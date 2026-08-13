"""Thread-safe state coordination for runtime camera controls."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

from ..config import load_camera_control_config
from ..controls import build_argus_control_properties, coerce_enum
from ..models import (
    CameraCapabilities,
    CameraControlUpdate,
    CameraProfile,
    CameraSettings,
    DenoiseMode,
    ProfileNotFoundError,
    RuntimeHandler,
    UnsupportedControlError,
    WhiteBalanceMode,
    capabilities_to_dict,
    settings_to_dict,
)

UNSET: Final[object] = object()


class CameraController:
    """Coordinate atomic runtime updates and in-memory camera profiles.

    The controller owns desired state, validation, and profile management. A
    runtime handler applies the resulting source properties before the new
    state is committed, preserving atomic behavior when the camera rejects an
    update.

    Args:
        capabilities: Properties and mode metadata for this camera. When
            omitted, the value from ``config.yml`` is used.
        initial_settings: State before the first update. When omitted, the
            value from ``config.yml`` is used.
        config_path: Optional camera-control YAML configuration path.
        runtime_handler: Callback that applies a validated pipeline update.
    """

    def __init__(
        self,
        capabilities: CameraCapabilities | None = None,
        initial_settings: CameraSettings | None = None,
        *,
        config_path: str | Path | None = None,
        runtime_handler: RuntimeHandler | None = None,
    ) -> None:
        """Initialize the controller without opening a camera device."""
        config = load_camera_control_config(config_path)
        resolved_capabilities = capabilities or config.capabilities
        resolved_initial_settings = initial_settings or config.initial_settings
        build_argus_control_properties(
            resolved_initial_settings,
            resolved_capabilities,
        )
        self._capabilities = resolved_capabilities
        self._settings = resolved_initial_settings
        self._runtime_handler = runtime_handler
        self._profiles: dict[str, CameraProfile] = {}
        self._revision = 0
        self._lock = threading.RLock()

    @property
    def capabilities(self) -> CameraCapabilities:
        """CameraCapabilities: Immutable capabilities declared for this camera."""
        return self._capabilities

    @property
    def settings(self) -> CameraSettings:
        """CameraSettings: A consistent snapshot of the desired camera state."""
        with self._lock:
            return self._settings

    @property
    def revision(self) -> int:
        """Return the monotonically increasing current-settings version."""
        with self._lock:
            return self._revision

    def get_runtime_state(self) -> dict[str, object]:
        """Return JSON-ready settings, capabilities, and profile metadata."""
        with self._lock:
            return {
                "revision": self._revision,
                "settings": settings_to_dict(self._settings),
                "capabilities": capabilities_to_dict(self._capabilities),
                "profiles": sorted(self._profiles),
                "source_properties": list(
                    build_argus_control_properties(self._settings, self._capabilities)
                ),
                "restart_required": False,
            }

    def set_runtime_handler(self, runtime_handler: RuntimeHandler | None) -> None:
        """Set or clear the callback that applies future runtime updates."""
        with self._lock:
            self._runtime_handler = runtime_handler

    def update(
        self,
        *,
        exposure_us: int | None | object = UNSET,
        gain: float | None | object = UNSET,
        awb_mode: WhiteBalanceMode | str | object = UNSET,
        awb_locked: bool | object = UNSET,
        denoise_mode: DenoiseMode | str | object = UNSET,
        denoise_strength: float | None | object = UNSET,
        sensor_mode: int | None | object = UNSET,
        hdr_enabled: bool | object = UNSET,
    ) -> CameraControlUpdate:
        """Apply one atomic settings update through the runtime handler."""
        with self._lock:
            settings = CameraSettings(
                exposure_us=(
                    self._settings.exposure_us
                    if exposure_us is UNSET
                    else cast(int | None, exposure_us)
                ),
                gain=(
                    self._settings.gain if gain is UNSET else cast(float | None, gain)
                ),
                awb_mode=(
                    self._settings.awb_mode
                    if awb_mode is UNSET
                    else cast(
                        WhiteBalanceMode,
                        coerce_enum(WhiteBalanceMode, awb_mode, "awb_mode"),
                    )
                ),
                awb_locked=(
                    self._settings.awb_locked
                    if awb_locked is UNSET
                    else cast(bool, awb_locked)
                ),
                denoise_mode=(
                    self._settings.denoise_mode
                    if denoise_mode is UNSET
                    else cast(
                        DenoiseMode,
                        coerce_enum(DenoiseMode, denoise_mode, "denoise_mode"),
                    )
                ),
                denoise_strength=(
                    self._settings.denoise_strength
                    if denoise_strength is UNSET
                    else cast(float | None, denoise_strength)
                ),
                sensor_mode=(
                    self._settings.sensor_mode
                    if sensor_mode is UNSET
                    else cast(int | None, sensor_mode)
                ),
                hdr_enabled=(
                    self._settings.hdr_enabled
                    if hdr_enabled is UNSET
                    else cast(bool, hdr_enabled)
                ),
            )
            return self._commit(settings)

    def set_exposure(self, exposure_us: int | None) -> CameraControlUpdate:
        """Set a fixed exposure or return exposure to Argus automatic control."""
        return self.update(exposure_us=exposure_us)

    def set_gain(self, gain: float | None) -> CameraControlUpdate:
        """Set fixed analog gain or return gain to Argus automatic control."""
        return self.update(gain=gain)

    def set_awb(
        self,
        mode: WhiteBalanceMode | str = WhiteBalanceMode.AUTO,
        *,
        locked: bool = False,
    ) -> CameraControlUpdate:
        """Set the white-balance mode and lock state."""
        return self.update(awb_mode=mode, awb_locked=locked)

    def set_denoise(
        self,
        mode: DenoiseMode | str,
        *,
        strength: float | None = None,
    ) -> CameraControlUpdate:
        """Set temporal denoising mode and optional strength."""
        return self.update(denoise_mode=mode, denoise_strength=strength)

    def set_sensor_mode(self, sensor_mode: int | None) -> CameraControlUpdate:
        """Select a sensor mode or restore Argus automatic mode selection."""
        return self.update(sensor_mode=sensor_mode)

    def set_hdr(self, enabled: bool) -> CameraControlUpdate:
        """Select a matching declared native HDR or non-HDR sensor mode."""
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")

        with self._lock:
            modes = self._capabilities.sensor_modes
            if not modes:
                raise UnsupportedControlError(
                    "HDR selection requires declared sensor-mode capabilities"
                )
            current_mode = self._capabilities.get_sensor_mode(
                self._settings.sensor_mode
            )
            if enabled:
                selected = next((mode for mode in modes if mode.hdr), None)
            elif current_mode is not None and current_mode.hdr:
                selected = next((mode for mode in modes if not mode.hdr), None)
            else:
                return self._commit(replace(self._settings, hdr_enabled=False))

            if selected is None:
                requested = "HDR" if enabled else "non-HDR"
                raise UnsupportedControlError(
                    f"no declared {requested} sensor mode is available"
                )
            return self._commit(
                replace(
                    self._settings,
                    sensor_mode=selected.index,
                    hdr_enabled=enabled,
                )
            )

    def save_profile(
        self,
        name: str,
        settings: CameraSettings | None = None,
    ) -> CameraProfile:
        """Store a validated named in-memory profile."""
        with self._lock:
            profile = CameraProfile(name=name, settings=settings or self._settings)
            build_argus_control_properties(profile.settings, self._capabilities)
            self._profiles[name] = profile
            return profile

    def get_profile(self, name: str) -> CameraProfile:
        """Return one named profile.

        Raises:
            ProfileNotFoundError: If the profile is absent.
        """
        with self._lock:
            try:
                return self._profiles[name]
            except KeyError as error:
                raise ProfileNotFoundError(name) from error

    def list_profiles(self) -> tuple[CameraProfile, ...]:
        """Return all profiles ordered by name."""
        with self._lock:
            return tuple(self._profiles[name] for name in sorted(self._profiles))

    def delete_profile(self, name: str) -> None:
        """Delete one named profile.

        Raises:
            ProfileNotFoundError: If the profile is absent.
        """
        with self._lock:
            if name not in self._profiles:
                raise ProfileNotFoundError(name)
            del self._profiles[name]

    def apply_profile(self, name: str) -> CameraControlUpdate:
        """Apply one stored profile through the runtime handler."""
        with self._lock:
            return self._commit(self.get_profile(name).settings)

    def _commit(self, settings: CameraSettings) -> CameraControlUpdate:
        """Validate, apply, and commit one settings transition under a lock."""
        source_properties = build_argus_control_properties(settings, self._capabilities)
        changed_fields = tuple(
            field_name
            for field_name in settings.__dataclass_fields__
            if getattr(self._settings, field_name) != getattr(settings, field_name)
        )
        manual_control_reset = (
            self._settings.exposure_us is not None and settings.exposure_us is None
        ) or (self._settings.gain is not None and settings.gain is None)
        update = CameraControlUpdate(
            revision=self._revision + (1 if changed_fields else 0),
            previous=self._settings,
            settings=settings,
            changed_fields=changed_fields,
            source_properties=source_properties,
            restart_required=bool(
                {"sensor_mode", "hdr_enabled"}.intersection(changed_fields)
            )
            or manual_control_reset,
        )
        if changed_fields and self._runtime_handler is not None:
            self._runtime_handler(update)
        if changed_fields:
            self._settings = settings
            self._revision = update.revision
        return update
