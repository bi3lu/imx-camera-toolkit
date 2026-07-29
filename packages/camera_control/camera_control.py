"""Runtime controls for NVIDIA Argus CSI camera pipelines.

The module is deliberately independent of a particular web framework and of
camera acquisition.  It models the controls accepted by ``nvarguscamerasrc``,
validates them against declared sensor capabilities, and emits an atomic
runtime update whenever a setting changes.

The project's GStreamer camera backend applies exposure, gain, white balance,
and temporal denoising directly to the active Argus source. Selecting a sensor
mode or HDR mode remains a pipeline-level change and is marked accordingly in
each runtime update.
"""

from __future__ import annotations

import re
import subprocess
import threading

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Callable, Final, Iterable, Sequence


class DenoiseMode(str, Enum):
    """Temporal noise-reduction modes provided by ``nvarguscamerasrc``."""

    OFF = "off"
    FAST = "fast"
    HIGH_QUALITY = "high_quality"


class WhiteBalanceMode(str, Enum):
    """White-balance modes provided by NVIDIA Argus."""

    OFF = "off"
    AUTO = "auto"
    INCANDESCENT = "incandescent"
    FLUORESCENT = "fluorescent"
    WARM_FLUORESCENT = "warm-fluorescent"
    DAYLIGHT = "daylight"
    CLOUDY_DAYLIGHT = "cloudy-daylight"
    TWILIGHT = "twilight"
    SHADE = "shade"
    MANUAL = "manual"


class UnsupportedControlError(ValueError):
    """Raised when a requested control is unavailable on the active sensor."""


class ProfileNotFoundError(KeyError):
    """Raised when a requested in-memory camera profile does not exist."""


@dataclass(frozen=True)
class SensorMode:
    """One sensor mode made available by a sensor driver.

    Args:
        index: Argus sensor-mode index passed to ``nvarguscamerasrc``.
        width: Native frame width in pixels, when known.
        height: Native frame height in pixels, when known.
        max_fps: Maximum frame rate, when known.
        hdr: Whether this mode is an HDR or WDR sensor mode.
        name: Optional human-readable identifier.
    """

    index: int
    width: int | None = None
    height: int | None = None
    max_fps: float | None = None
    hdr: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate a sensor-mode descriptor."""
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ValueError("sensor mode index must be an integer")

        if not 0 <= self.index <= 255:
            raise ValueError("sensor mode index must be between 0 and 255")

        for field_name in ("width", "height"):
            value = getattr(self, field_name)

            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"sensor mode {field_name} must be a positive integer")

        if self.max_fps is not None and (
            isinstance(self.max_fps, bool)
            or not isinstance(self.max_fps, (int, float))
            or self.max_fps <= 0
        ):
            raise ValueError("sensor mode max_fps must be a positive number")

        if self.name is not None and not self.name.strip():
            raise ValueError("sensor mode name must not be empty")


@dataclass(frozen=True)
class CameraCapabilities:
    """Controls and sensor modes known to be supported by a camera.

    ``sensor_modes`` is intentionally supplied by the application or sensor
    integration.  Argus exposes the number of modes at runtime but does not
    expose complete mode metadata through ``gst-inspect-1.0``.  An empty tuple
    means that mode metadata is unknown, not that the sensor has no modes.

    Args:
        source_properties: GStreamer properties exposed by the Argus source.
        sensor_modes: Known sensor modes for the selected sensor.
        model: Optional sensor model, for example ``"IMX219"``.
    """

    source_properties: frozenset[str]
    sensor_modes: tuple[SensorMode, ...] = ()
    model: str | None = None

    def __post_init__(self) -> None:
        """Normalise and validate capability metadata."""
        properties = frozenset(
            property_name.strip().lower()
            for property_name in self.source_properties
            if property_name.strip()
        )
        object.__setattr__(self, "source_properties", properties)

        indices = [mode.index for mode in self.sensor_modes]

        if len(indices) != len(set(indices)):
            raise ValueError("sensor mode indices must be unique")

        if self.model is not None and not self.model.strip():
            raise ValueError("camera model must not be empty")

    @property
    def hdr_supported(self) -> bool:
        """bool: Whether declared sensor modes include an HDR mode."""
        return any(mode.hdr for mode in self.sensor_modes)

    @property
    def sensor_mode_metadata_available(self) -> bool:
        """bool: Whether complete mode descriptors were supplied."""
        return bool(self.sensor_modes)

    def supports(self, property_name: str) -> bool:
        """Return whether an Argus source property is known to be available.

        Args:
            property_name: Name as reported by ``gst-inspect-1.0``.

        Returns:
            ``True`` when the property is included in ``source_properties``.
        """
        return property_name.lower() in self.source_properties

    def get_sensor_mode(self, index: int) -> SensorMode | None:
        """Return known metadata for a sensor mode.

        Args:
            index: Argus sensor-mode index.

        Returns:
            The matching descriptor, or ``None`` when mode metadata is absent.
        """
        return next((mode for mode in self.sensor_modes if mode.index == index), None)


ARGUS_CONTROL_PROPERTIES: Final[frozenset[str]] = frozenset(
    {
        "aelock",
        "awblock",
        "exposuretimerange",
        "gainrange",
        "ispdigitalgainrange",
        "sensor-mode",
        "tnr-mode",
        "tnr-strength",
        "wbmode",
    }
)

DEFAULT_CAPABILITIES: Final[CameraCapabilities] = CameraCapabilities(
    source_properties=ARGUS_CONTROL_PROPERTIES
)


@dataclass(frozen=True)
class CameraSettings:
    """Validated desired state for one NVIDIA Argus camera.

    Exposure is expressed in microseconds for the public Python API and is
    converted to nanoseconds for the ``exposuretimerange`` GStreamer property.
    Setting an exposure or analog gain pins the corresponding Argus range to
    one value and enables the auto-exposure lock.  Clearing both returns the
    sensor to automatic exposure.

    Args:
        exposure_us: Fixed exposure duration in microseconds, or ``None``.
        gain: Fixed analog gain, or ``None`` for automatic gain.
        awb_mode: Desired white-balance mode.
        awb_locked: Whether auto white balance is locked.
        denoise_mode: Temporal noise-reduction quality mode.
        denoise_strength: Denoising strength from 0.0 to 1.0, or ``None`` for
            the Argus default.
        sensor_mode: Sensor mode index, or ``None`` for Argus auto-selection.
        hdr_enabled: Whether an HDR sensor mode is requested.
    """

    exposure_us: int | None = None
    gain: float | None = None
    awb_mode: WhiteBalanceMode = WhiteBalanceMode.AUTO
    awb_locked: bool = False
    denoise_mode: DenoiseMode = DenoiseMode.FAST
    denoise_strength: float | None = None
    sensor_mode: int | None = None
    hdr_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate independent setting ranges and types."""
        if self.exposure_us is not None and (
            isinstance(self.exposure_us, bool)
            or not isinstance(self.exposure_us, int)
            or self.exposure_us <= 0
        ):
            raise ValueError("exposure_us must be a positive integer or None")

        if self.gain is not None and (
            isinstance(self.gain, bool)
            or not isinstance(self.gain, (int, float))
            or self.gain <= 0
        ):
            raise ValueError("gain must be a positive number or None")

        if not isinstance(self.awb_locked, bool):
            raise ValueError("awb_locked must be a boolean")

        if not isinstance(self.awb_mode, WhiteBalanceMode):
            raise ValueError("awb_mode must be a WhiteBalanceMode")

        if not isinstance(self.denoise_mode, DenoiseMode):
            raise ValueError("denoise_mode must be a DenoiseMode")

        if self.denoise_strength is not None and (
            isinstance(self.denoise_strength, bool)
            or not isinstance(self.denoise_strength, (int, float))
            or not 0.0 <= self.denoise_strength <= 1.0
        ):
            raise ValueError("denoise_strength must be between 0.0 and 1.0")

        if self.sensor_mode is not None and (
            isinstance(self.sensor_mode, bool)
            or not isinstance(self.sensor_mode, int)
            or not 0 <= self.sensor_mode <= 255
        ):
            raise ValueError("sensor_mode must be between 0 and 255 or None")

        if not isinstance(self.hdr_enabled, bool):
            raise ValueError("hdr_enabled must be a boolean")


@dataclass(frozen=True)
class CameraProfile:
    """A named, in-memory snapshot of camera settings.

    Profiles are deliberately not persisted here.  Persistence belongs to the
    future configuration layer and can serialise this small immutable model.

    Args:
        name: Unique profile identifier.
        settings: Settings restored when the profile is applied.
    """

    name: str
    settings: CameraSettings

    def __post_init__(self) -> None:
        """Validate the profile name."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.name):
            raise ValueError(
                "profile name must be 1-64 characters containing only letters, "
                "numbers, dots, underscores, and hyphens"
            )

        if not isinstance(self.settings, CameraSettings):
            raise ValueError("profile settings must be a CameraSettings instance")


@dataclass(frozen=True)
class CameraControlUpdate:
    """One atomic camera-control state transition.

    Args:
        revision: Monotonically increasing state version.
        previous: Settings before the update.
        settings: Settings requested by the update.
        changed_fields: Names of fields whose values changed.
        source_properties: Properties to append to ``nvarguscamerasrc``.
        restart_required: Whether the current capture backend must restart.
    """

    revision: int
    previous: CameraSettings
    settings: CameraSettings
    changed_fields: tuple[str, ...]
    source_properties: tuple[str, ...]
    restart_required: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation for a future API layer.

        Returns:
            Update state using enum values instead of enum instances.
        """
        return {
            "revision": self.revision,
            "previous": _settings_to_dict(self.previous),
            "settings": _settings_to_dict(self.settings),
            "changed_fields": list(self.changed_fields),
            "source_properties": list(self.source_properties),
            "restart_required": self.restart_required,
        }


RuntimeHandler = Callable[[CameraControlUpdate], None]
UNSET: Final[object] = object()


def discover_argus_capabilities(
    command: Sequence[str] = ("gst-inspect-1.0", "nvarguscamerasrc"),
    *,
    timeout: float = 3.0,
    sensor_modes: Iterable[SensorMode] = (),
    model: str | None = None,
) -> CameraCapabilities:
    """Inspect the locally installed Argus source without opening a sensor.

    Args:
        command: Command used to inspect the GStreamer element.
        timeout: Maximum execution time in seconds.
        sensor_modes: Optional sensor-mode metadata supplied by the integrator.
        model: Optional sensor model associated with the supplied modes.

    Returns:
        Discovered source property names and supplied sensor metadata.

    Raises:
        RuntimeError: If GStreamer inspection is unavailable or unsuccessful.
        ValueError: If the timeout is invalid.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive number")

    if timeout <= 0:
        raise ValueError("timeout must be a positive number")

    if not command:
        raise ValueError("inspection command must not be empty")

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not inspect nvarguscamerasrc") from error

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not inspect nvarguscamerasrc: {message}")

    property_pattern = r"^  ([a-z][a-z0-9-]*)\s+:"
    properties = frozenset(
        match.group(1)
        for match in re.finditer(property_pattern, result.stdout, re.MULTILINE)
    )
    if not properties:
        raise RuntimeError("nvarguscamerasrc did not report any properties")

    return CameraCapabilities(
        source_properties=properties,
        sensor_modes=tuple(sensor_modes),
        model=model,
    )


def build_argus_control_properties(
    settings: CameraSettings,
    capabilities: CameraCapabilities = DEFAULT_CAPABILITIES,
) -> tuple[str, ...]:
    """Build safe ``nvarguscamerasrc`` property assignments from settings.

    Args:
        settings: Desired camera state.
        capabilities: Properties and modes supported by the selected camera.

    Returns:
        Ordered GStreamer property assignments, for example
        ``('exposuretimerange="10000000 10000000"', 'aelock=true')``.

    Raises:
        UnsupportedControlError: If the camera cannot provide a requested
            feature or sensor mode.
    """
    _validate_settings_capabilities(settings, capabilities)
    properties: list[str] = []

    if settings.sensor_mode is not None:
        properties.append(f"sensor-mode={settings.sensor_mode}")

    exposure_or_gain_fixed = (
        settings.exposure_us is not None or settings.gain is not None
    )

    if settings.exposure_us is not None:
        exposure_ns = settings.exposure_us * 1_000
        properties.append(f'exposuretimerange="{exposure_ns} {exposure_ns}"')

    if settings.gain is not None:
        gain = _format_number(settings.gain)
        properties.append(f'gainrange="{gain} {gain}"')

    if exposure_or_gain_fixed:
        properties.append('ispdigitalgainrange="1 1"')
        properties.append("aelock=false")

    properties.append(f"wbmode={settings.awb_mode.value}")
    properties.append(f"awblock={_format_bool(settings.awb_locked)}")
    properties.append(f"tnr-mode={_denoise_mode_value(settings.denoise_mode)}")

    if settings.denoise_strength is not None:
        properties.append(f"tnr-strength={_format_number(settings.denoise_strength)}")

    return tuple(properties)


class CameraController:
    """Thread-safe runtime API for Argus camera settings and profiles.

    The controller owns only desired state.  A supplied ``runtime_handler`` is
    invoked before the new state is committed; it applies
    ``update.source_properties`` to the active capture pipeline. If that
    operation fails, the controller retains its previous state and profiles.

    Args:
        capabilities: Properties and sensor-mode metadata for this camera.
        initial_settings: Desired state used before the first update.
        runtime_handler: Callback that applies a validated pipeline update.

    Raises:
        UnsupportedControlError: If initial settings are unsupported.
    """

    def __init__(
        self,
        capabilities: CameraCapabilities = DEFAULT_CAPABILITIES,
        initial_settings: CameraSettings = CameraSettings(),
        *,
        runtime_handler: RuntimeHandler | None = None,
    ) -> None:
        """Initialize the controller without opening a camera device."""
        build_argus_control_properties(initial_settings, capabilities)
        self._capabilities = capabilities
        self._settings = initial_settings
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
        """int: Monotonically increasing version of the current settings."""
        with self._lock:
            return self._revision

    def get_runtime_state(self) -> dict[str, object]:
        """Return JSON-ready settings, capabilities, and profile metadata.

        Returns:
            A stable snapshot suitable for a future HTTP runtime API.
        """
        with self._lock:
            return {
                "revision": self._revision,
                "settings": _settings_to_dict(self._settings),
                "capabilities": _capabilities_to_dict(self._capabilities),
                "profiles": sorted(self._profiles),
                "source_properties": list(
                    build_argus_control_properties(self._settings, self._capabilities)
                ),
                "restart_required": False,
            }

    def set_runtime_handler(self, runtime_handler: RuntimeHandler | None) -> None:
        """Set or clear the callback that applies future runtime updates.

        Args:
            runtime_handler: Callback receiving each validated update, or
                ``None`` to manage desired state without touching a camera.
        """
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
        """Apply one atomic settings update through the runtime handler.

        Passing ``None`` to ``exposure_us``, ``gain``, ``denoise_strength``,
        or ``sensor_mode`` explicitly restores its automatic/default state.
        Omitting a keyword preserves its current value.

        Args:
            exposure_us: Fixed exposure time in microseconds, or ``None``.
            gain: Fixed analog gain, or ``None``.
            awb_mode: White-balance mode.
            awb_locked: Auto-white-balance lock state.
            denoise_mode: Temporal denoising mode.
            denoise_strength: Strength from 0.0 to 1.0, or ``None``.
            sensor_mode: Explicit Argus mode index, or ``None``.
            hdr_enabled: Select an HDR sensor mode when enabled.

        Returns:
            The committed transition. An unchanged request has no new revision.

        Raises:
            UnsupportedControlError: If a requested feature is unavailable.
            ValueError: If a setting value is invalid.
        """
        values: dict[str, object] = {}
        for field_name, value in {
            "exposure_us": exposure_us,
            "gain": gain,
            "awb_mode": awb_mode,
            "awb_locked": awb_locked,
            "denoise_mode": denoise_mode,
            "denoise_strength": denoise_strength,
            "sensor_mode": sensor_mode,
            "hdr_enabled": hdr_enabled,
        }.items():
            if value is not UNSET:
                values[field_name] = value

        if "awb_mode" in values:
            values["awb_mode"] = _coerce_enum(
                WhiteBalanceMode, values["awb_mode"], "awb_mode"
            )
        if "denoise_mode" in values:
            values["denoise_mode"] = _coerce_enum(
                DenoiseMode, values["denoise_mode"], "denoise_mode"
            )

        with self._lock:
            return self._commit(replace(self._settings, **values))

    def set_exposure(self, exposure_us: int | None) -> CameraControlUpdate:
        """Set a fixed exposure or return exposure to Argus automatic control.

        Args:
            exposure_us: Exposure duration in microseconds, or ``None``.

        Returns:
            The committed runtime transition.
        """
        return self.update(exposure_us=exposure_us)

    def set_gain(self, gain: float | None) -> CameraControlUpdate:
        """Set fixed analog gain or return gain to Argus automatic control.

        Args:
            gain: Positive analog gain, or ``None``.

        Returns:
            The committed runtime transition.
        """
        return self.update(gain=gain)

    def set_awb(
        self,
        mode: WhiteBalanceMode | str = WhiteBalanceMode.AUTO,
        *,
        locked: bool = False,
    ) -> CameraControlUpdate:
        """Set the white-balance mode and lock state.

        Args:
            mode: White-balance mode accepted by Argus.
            locked: Whether to lock auto white balance.

        Returns:
            The committed runtime transition.
        """
        return self.update(awb_mode=mode, awb_locked=locked)

    def set_denoise(
        self,
        mode: DenoiseMode | str,
        *,
        strength: float | None = None,
    ) -> CameraControlUpdate:
        """Set temporal denoising mode and optional strength.

        Args:
            mode: Denoising mode accepted by Argus.
            strength: Strength from 0.0 to 1.0, or ``None`` for the driver
                default.

        Returns:
            The committed runtime transition.
        """
        return self.update(denoise_mode=mode, denoise_strength=strength)

    def set_sensor_mode(self, sensor_mode: int | None) -> CameraControlUpdate:
        """Select a sensor mode or restore Argus automatic mode selection.

        Args:
            sensor_mode: Argus sensor-mode index, or ``None``.

        Returns:
            The committed runtime transition.
        """
        return self.update(sensor_mode=sensor_mode)

    def set_hdr(self, enabled: bool) -> CameraControlUpdate:
        """Enable or disable HDR by selecting a matching declared sensor mode.

        Args:
            enabled: ``True`` selects the first declared HDR mode. ``False``
                selects the first declared non-HDR mode when the current mode
                is HDR; otherwise it leaves the mode unchanged.

        Returns:
            The committed runtime transition.

        Raises:
            UnsupportedControlError: If sensor-mode metadata cannot select the
                requested HDR state.
        """
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
        self, name: str, settings: CameraSettings | None = None
    ) -> CameraProfile:
        """Store a named in-memory profile after validating its capabilities.

        Args:
            name: Profile identifier.
            settings: Explicit settings, or the current state when omitted.

        Returns:
            The stored immutable profile.
        """
        with self._lock:
            profile = CameraProfile(name=name, settings=settings or self._settings)
            build_argus_control_properties(profile.settings, self._capabilities)
            self._profiles[name] = profile
            return profile

    def get_profile(self, name: str) -> CameraProfile:
        """Return one named profile.

        Args:
            name: Profile identifier.

        Returns:
            The requested immutable profile.

        Raises:
            ProfileNotFoundError: If the profile is absent.
        """
        with self._lock:
            try:
                return self._profiles[name]

            except KeyError as error:
                raise ProfileNotFoundError(name) from error

    def list_profiles(self) -> tuple[CameraProfile, ...]:
        """Return profiles ordered by their name.

        Returns:
            All in-memory profiles in stable order.
        """
        with self._lock:
            return tuple(self._profiles[name] for name in sorted(self._profiles))

    def delete_profile(self, name: str) -> None:
        """Delete one in-memory profile.

        Args:
            name: Profile identifier.

        Raises:
            ProfileNotFoundError: If the profile is absent.
        """
        with self._lock:
            if name not in self._profiles:
                raise ProfileNotFoundError(name)

            del self._profiles[name]

    def apply_profile(self, name: str) -> CameraControlUpdate:
        """Apply one stored profile through the runtime handler.

        Args:
            name: Profile identifier.

        Returns:
            The committed runtime transition.

        Raises:
            ProfileNotFoundError: If the profile is absent.
        """
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


def _validate_settings_capabilities(
    settings: CameraSettings, capabilities: CameraCapabilities
) -> None:
    """Validate settings that depend on available source properties."""
    required_properties = {"wbmode", "awblock", "tnr-mode"}
    if settings.sensor_mode is not None:
        required_properties.add("sensor-mode")

    if settings.exposure_us is not None:
        required_properties.add("exposuretimerange")

    if settings.gain is not None:
        required_properties.add("gainrange")

    if settings.exposure_us is not None or settings.gain is not None:
        required_properties.add("aelock")
        required_properties.add("ispdigitalgainrange")

    if settings.denoise_strength is not None:
        required_properties.add("tnr-strength")

    missing = sorted(
        property_name
        for property_name in required_properties
        if not capabilities.supports(property_name)
    )
    if missing:
        raise UnsupportedControlError(
            "camera does not support required control(s): " + ", ".join(missing)
        )

    if settings.sensor_mode is not None and capabilities.sensor_mode_metadata_available:
        mode = capabilities.get_sensor_mode(settings.sensor_mode)

        if mode is None:
            raise UnsupportedControlError(
                f"sensor mode {settings.sensor_mode} is not declared by capabilities"
            )

        if mode.hdr != settings.hdr_enabled:
            expected = "HDR" if settings.hdr_enabled else "non-HDR"
            raise UnsupportedControlError(
                f"sensor mode {settings.sensor_mode} is not a declared {expected} mode"
            )

    if settings.hdr_enabled:
        if not capabilities.hdr_supported:
            raise UnsupportedControlError("camera does not declare an HDR sensor mode")

        if settings.sensor_mode is None:
            raise UnsupportedControlError("HDR requires an explicit HDR sensor mode")


def _coerce_enum(enum_type: type[Enum], value: object, field_name: str) -> Enum:
    """Convert a string or enum instance to one supported enum value."""
    if isinstance(value, enum_type):
        return value

    if isinstance(value, str):
        try:
            return enum_type(value)

        except ValueError as error:
            choices = ", ".join(member.value for member in enum_type)
            raise ValueError(f"{field_name} must be one of: {choices}") from error

    raise ValueError(f"{field_name} must be a {enum_type.__name__} or string")


def _denoise_mode_value(mode: DenoiseMode) -> int:
    """Return the integer representation used by ``tnr-mode``."""
    return {
        DenoiseMode.OFF: 0,
        DenoiseMode.FAST: 1,
        DenoiseMode.HIGH_QUALITY: 2,
    }[mode]


def _format_bool(value: bool) -> str:
    """Format a boolean as a GStreamer-compatible literal."""
    return "true" if value else "false"


def _format_number(value: int | float) -> str:
    """Format a numeric property without scientific notation or trailing zeroes."""
    return format(value, ".15g")


def _settings_to_dict(settings: CameraSettings) -> dict[str, object]:
    """Convert immutable settings to JSON-ready primitive values."""
    values = asdict(settings)
    values["awb_mode"] = settings.awb_mode.value
    values["denoise_mode"] = settings.denoise_mode.value
    return values


def _capabilities_to_dict(capabilities: CameraCapabilities) -> dict[str, object]:
    """Convert capabilities to a JSON-ready representation."""
    return {
        "model": capabilities.model,
        "source_properties": sorted(capabilities.source_properties),
        "hdr_supported": capabilities.hdr_supported,
        "sensor_mode_metadata_available": capabilities.sensor_mode_metadata_available,
        "sensor_modes": [asdict(mode) for mode in capabilities.sensor_modes],
    }
