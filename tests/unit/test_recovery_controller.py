"""Tests for the backend-independent recovery budget."""

import pytest

from imx_camera_toolkit._internal.camera.recovery import RecoveryController


@pytest.mark.parametrize("max_attempts", [0, 1, 3])
def test_budget_is_bounded_and_renews_after_a_frame(max_attempts: int) -> None:
    """Admission is bounded across calls and only a frame renews the budget."""
    controller = RecoveryController()

    for _ in range(2):
        assert controller.attempts == 0

        for attempt in range(1, max_attempts + 1):
            assert controller.begin_attempt(max_attempts) == attempt

        assert controller.begin_attempt(max_attempts) is None
        assert controller.begin_attempt(max_attempts) is None
        assert controller.attempts == max_attempts
        controller.record_frame_success()


def test_diagnostics_do_not_change_budget() -> None:
    """Camera-owned error reporting must neither spend nor renew attempts."""
    controller = RecoveryController()
    failure = RuntimeError("reopen failed")
    assert controller.last_error is None
    assert controller.begin_attempt(1) == 1
    controller.last_error = failure
    assert controller.begin_attempt(1) is None
    controller.last_error = None
    assert controller.begin_attempt(1) is None
    controller.last_error = failure
    controller.record_frame_success()
    assert controller.last_error is failure
    assert controller.begin_attempt(1) == 1
