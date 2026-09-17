"""Empirical adversarial stress test for DarwinIdleMonitor power and battery isolation.

This script challenges DarwinIdleMonitor across:
1. Rapidly alternating power states (AC, Battery, Desktop/None, Pathological values).
2. Concurrent multi-threaded query stress with per-thread state isolation.
3. Boundary, malformed, and corrupted sensor inputs.
4. IOKit and cross-platform fallback isolation.
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass
from unittest import mock

from quanta.cognitive.darwin_idle import DarwinIdleMonitor, PowerTelemetry


@dataclass
class MockBattery:
    percent: float | int
    secsleft: int | None
    power_plugged: bool


def test_rapid_alternating_power_states(iterations: int = 1000) -> None:
    """Stress test rapid transitions between AC, Battery, and Absent states."""
    monitor = DarwinIdleMonitor()
    states = [
        # (plugged, percent, secsleft, expected_on_battery, expected_source_str)
        (True, 100, None, False, "AC Power"),
        (False, 85, 7200, True, "Battery Power"),
        (False, 15, 900, True, "Battery Power"),
        (True, 45, None, False, "AC Power"),
        (None, None, None, False, "AC Power"),  # Sensor absent (desktop)
    ]

    for i in range(iterations):
        plugged, pct, secs, exp_bat, exp_str = states[i % len(states)]
        if plugged is None:
            mock_val = None
        else:
            mock_val = MockBattery(percent=pct, secsleft=secs, power_plugged=plugged)

        # Bypass Darwin IOKit to stress the fallback / cross-platform path directly
        with (
            mock.patch("psutil.sensors_battery", return_value=mock_val),
            mock.patch.object(monitor, "_iokit", None),
        ):
            pwr = monitor.get_power_telemetry()
            on_batt = monitor.is_on_battery()
            quiescence = monitor.assess_quiescence()

            assert on_batt is exp_bat, (
                f"Iteration {i}: is_on_battery() was {on_batt}, expected {exp_bat}"
            )
            assert quiescence.power_source == exp_str, (
                f"Iteration {i}: power_source was {quiescence.power_source}, expected {exp_str}"
            )
            if plugged is not None:
                assert pwr.battery_level_pct == pct
                assert pwr.is_ac_powered is plugged
            else:
                assert pwr.is_ac_powered is True
                assert pwr.battery_level_pct is None

    print(f"✓ Passed rapid alternating power states ({iterations} iterations)")


def test_pathological_and_corrupt_sensor_inputs() -> None:
    """Stress test boundary, malformed, and exception-throwing sensor conditions."""
    monitor = DarwinIdleMonitor()

    pathological_cases = [
        # Edge percent boundaries
        MockBattery(percent=0, secsleft=0, power_plugged=False),
        MockBattery(percent=100, secsleft=-1, power_plugged=True),
        MockBattery(percent=-5, secsleft=None, power_plugged=True),
        MockBattery(percent=150, secsleft=None, power_plugged=False),
        # None secsleft
        MockBattery(percent=50, secsleft=None, power_plugged=False),
    ]

    for case in pathological_cases:
        with (
            mock.patch("psutil.sensors_battery", return_value=case),
            mock.patch.object(monitor, "_iokit", None),
        ):
            pwr = monitor.get_power_telemetry()
            on_batt = monitor.is_on_battery()
            quiescence = monitor.assess_quiescence()

            assert isinstance(on_batt, bool)
            assert isinstance(pwr, PowerTelemetry)
            assert quiescence.power_source in ("AC Power", "Battery Power")
            assert on_batt is (not case.power_plugged)

    # Test sensor throwing severe exceptions
    exceptions_to_test = [
        PermissionError("Access denied to IOKit / power supply"),
        OSError("Device I/O error"),
        RuntimeError("Kernel subsystem busy"),
        AttributeError("Malformed battery object"),
    ]

    for exc in exceptions_to_test:
        with (
            mock.patch("psutil.sensors_battery", side_effect=exc),
            mock.patch.object(monitor, "_iokit", None),
        ):
            pwr = monitor.get_power_telemetry()
            on_batt = monitor.is_on_battery()
            quiescence = monitor.assess_quiescence()

            # In catastrophic failure, must safely default to AC Desktop
            assert on_batt is False
            assert pwr.is_ac_powered is True
            assert pwr.battery_level_pct is None
            assert pwr.battery_health == "Unknown"
            assert quiescence.power_source == "AC Power"

    print("✓ Passed pathological and corrupt sensor inputs")


def test_concurrent_multithreaded_power_queries(
    num_workers: int = 12, calls_per_worker: int = 150
) -> None:
    """Stress test concurrent access across threads with thread-safe mock isolation."""
    monitor = DarwinIdleMonitor()
    thread_states: dict[int, MockBattery | None] = {}
    lock = threading.Lock()

    def mock_sensors_battery() -> MockBattery | None:
        with lock:
            return thread_states.get(threading.get_ident())

    def worker_routine(worker_id: int) -> int:
        for step in range(calls_per_worker):
            is_ac = (worker_id + step) % 2 == 0
            val = MockBattery(
                percent=20 + (step % 80),
                secsleft=1800 if not is_ac else None,
                power_plugged=is_ac,
            )
            with lock:
                thread_states[threading.get_ident()] = val

            telemetry = monitor.get_power_telemetry()
            on_battery = monitor.is_on_battery()
            state = monitor.assess_quiescence()

            assert on_battery is (not is_ac)
            assert telemetry.is_ac_powered is is_ac
            assert state.power_source == ("AC Power" if is_ac else "Battery Power")
        return calls_per_worker

    with (
        mock.patch.object(monitor, "_iokit", None),
        mock.patch("psutil.sensors_battery", side_effect=mock_sensors_battery),
        concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor,
    ):
        futures = [executor.submit(worker_routine, w) for w in range(num_workers)]
        total_calls = sum(f.result() for f in futures)

    print(
        f"✓ Passed concurrent multi-threaded power queries "
        f"({total_calls} calls across {num_workers} threads)"
    )


def test_iokit_simulated_null_and_malformed_states() -> None:
    """Stress test Darwin IOKit Ctypes paths with simulated NULL references."""
    monitor = DarwinIdleMonitor()

    # Case A: IOPSCopyPowerSourcesInfo returns None
    mock_iokit = mock.MagicMock()
    mock_iokit.IOPSCopyPowerSourcesInfo.return_value = None

    with (
        mock.patch.object(monitor, "is_darwin", True),
        mock.patch.object(monitor, "_iokit", mock_iokit),
        mock.patch.object(monitor, "_cf", mock.MagicMock()),
        mock.patch("psutil.sensors_battery", return_value=None),
    ):
        telemetry = monitor.get_power_telemetry()
        assert telemetry.is_ac_powered is True
        assert telemetry.battery_level_pct is None
        assert monitor.is_on_battery() is False

    # Case B: IOPSCopyPowerSourcesList returns None (e.g. Mac Mini / Mac Pro without battery)
    mock_iokit.IOPSCopyPowerSourcesInfo.return_value = 0x1234
    mock_iokit.IOPSGetProvidingPowerSourceType.return_value = None
    mock_iokit.IOPSCopyPowerSourcesList.return_value = None
    mock_cf = mock.MagicMock()

    with (
        mock.patch.object(monitor, "is_darwin", True),
        mock.patch.object(monitor, "_iokit", mock_iokit),
        mock.patch.object(monitor, "_cf", mock_cf),
    ):
        telemetry = monitor.get_power_telemetry()
        assert telemetry.is_ac_powered is True
        assert telemetry.battery_level_pct is None
        assert telemetry.battery_health == "Unavailable"
        assert monitor.is_on_battery() is False

    print("✓ Passed IOKit simulated NULL and malformed states")


def main() -> None:
    print("============================================================")
    print("Running Empirical Adversarial Battery/AC Stress Test Suite")
    print("============================================================")
    test_rapid_alternating_power_states(iterations=1000)
    test_pathological_and_corrupt_sensor_inputs()
    test_concurrent_multithreaded_power_queries(num_workers=12, calls_per_worker=150)
    test_iokit_simulated_null_and_malformed_states()
    print("============================================================")
    print("All empirical battery vs AC isolation stress tests PASSED!")
    print("============================================================")


if __name__ == "__main__":
    main()
