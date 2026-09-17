"""Tests for Darwin Mach kernel QoS, hardware quiescence, and telemetry bindings.

Tests quanta.cognitive.darwin_idle against the interface contracts defined in
PROJECT.md, TEST_INFRA.md, and upstream survey reports.
Remediated by teamwork_preview_test_writer_test_track_gen2 to eliminate
self-certifying mock facades, add 32-bit counter overflow unit tests,
and verify low-level C subsystem fallbacks, error handling, and thread safety.
"""

from __future__ import annotations

import ctypes
import os
import platform
import threading
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest

from quanta.cognitive.darwin_idle import (
    QOS_CLASS_BACKGROUND,
    CpuTickSnapshot,
    DarwinIdleMonitor,
    HardwareQuiescenceState,
    PowerTelemetry,
    get_cpu_quiescence,
    get_default_monitor,
    get_thermal_state,
    is_on_battery,
    is_system_idle,
    set_background_qos,
)

# ============================================================================
# Tier 1: Real Host Hardware Execution Tests (when on macOS Darwin)
# ============================================================================


class TestDarwinHostHardware:
    """Validates real host execution on macOS Darwin or platform detection."""

    def test_platform_detection(self) -> None:
        """Verifies current platform detection logic."""
        current = platform.system()
        assert current in ("Darwin", "Linux", "Windows")

    @pytest.mark.skipif(platform.system() != "Darwin", reason="Darwin-specific hardware test")
    def test_real_set_background_qos_on_darwin(self) -> None:
        """Verifies setting background QoS returns a boolean without throwing."""
        result = set_background_qos()
        assert isinstance(result, bool)

    def test_real_get_cpu_quiescence_range(self) -> None:
        """Verifies CPU quiescence returns a valid probability ratio in [0.0, 1.0]."""
        ratio = get_cpu_quiescence()
        assert isinstance(ratio, (float, int))
        assert 0.0 <= ratio <= 1.0

    def test_real_get_thermal_state_range(self) -> None:
        """Verifies thermal state returns an integer in {0, 1, 2, 3}."""
        thermal = get_thermal_state()
        assert isinstance(thermal, int)
        assert thermal in (0, 1, 2, 3)

    def test_real_is_on_battery_type(self) -> None:
        """Verifies battery status returns a boolean."""
        on_battery = is_on_battery()
        assert isinstance(on_battery, bool)

    def test_real_is_system_idle_boolean(self) -> None:
        """Verifies system idle returns a boolean with default parameters."""
        idle = is_system_idle()
        assert isinstance(idle, bool)

    def test_real_assess_quiescence(self) -> None:
        """Verifies assess_quiescence returns a complete HardwareQuiescenceState."""
        monitor = get_default_monitor()
        state = monitor.assess_quiescence()
        assert isinstance(state, HardwareQuiescenceState)
        assert isinstance(state.is_quiescent, bool)
        assert 0.0 <= state.cpu_load_pct <= 100.0
        assert 0.0 <= state.cpu_idle_pct <= 100.0
        assert state.thermal_pressure_level in (0, 1, 2, 3)
        assert state.power_source in ("AC Power", "Battery Power")

    def test_real_compute_cpu_utilization(self) -> None:
        """Verifies compute_cpu_utilization returns (load_pct, idle_pct) summing to 100%."""
        monitor = get_default_monitor()
        load_pct, idle_pct = monitor.compute_cpu_utilization()
        assert 0.0 <= load_pct <= 100.0
        assert 0.0 <= idle_pct <= 100.0
        assert abs((load_pct + idle_pct) - 100.0) <= 0.05


# ============================================================================
# Tier 1: Boundary Value Analysis (BVA) for is_system_idle()
# ============================================================================


class TestSystemIdleBoundaryValues:
    """Boundary value analysis for is_system_idle across idle ratio and thermal state."""

    @pytest.mark.parametrize(
        ("idle_ratio", "threshold", "thermal", "max_thermal", "expected"),
        [
            # Just below idle threshold
            (0.699, 0.70, 0, 1, False),
            # Exactly at idle threshold
            (0.700, 0.70, 0, 1, True),
            # Just above idle threshold
            (0.701, 0.70, 0, 1, True),
            # Far above idle threshold
            (0.999, 0.70, 0, 1, True),
            # Thermal state exactly at max_thermal
            (0.850, 0.70, 1, 1, True),
            # Thermal state just above max_thermal (Serious)
            (0.850, 0.70, 2, 1, False),
            # Thermal state Critical
            (0.850, 0.70, 3, 1, False),
            # Custom strict idle threshold (0.90)
            (0.890, 0.90, 0, 1, False),
            (0.900, 0.90, 0, 1, True),
            (0.950, 0.90, 0, 1, True),
            # Custom strict thermal limit (max_thermal = 0: Nominal only)
            (0.850, 0.70, 1, 0, False),
            (0.850, 0.70, 0, 0, True),
            # Completely saturated system (0.0 idle)
            (0.000, 0.70, 0, 1, False),
            # Completely idle system (1.0 idle)
            (1.000, 0.70, 0, 1, True),
        ],
    )
    def test_is_system_idle_matrix(
        self,
        idle_ratio: float,
        threshold: float,
        thermal: int,
        max_thermal: int,
        expected: bool,
    ) -> None:
        """Validates all critical boundary transitions for system idle evaluation."""
        with (
            mock.patch.object(DarwinIdleMonitor, "get_cpu_quiescence", return_value=idle_ratio),
            mock.patch.object(DarwinIdleMonitor, "get_thermal_state", return_value=thermal),
        ):
            res = is_system_idle(idle_threshold=threshold, max_thermal=max_thermal)
            assert res == expected


# ============================================================================
# Tier 1: Ctypes & Mach Kernel Error Handling (No Facade Mocks)
# ============================================================================


class TestMachErrorHandling:
    """Verifies robustness against Mach IPC errors, NULL pointers, and kernel faults.

    All tests exercise the real implementation methods directly, injecting
    faults at the underlying C library, ctypes, or tick-generation boundary.
    """

    def test_cpu_quiescence_zero_total_ticks(self) -> None:
        """Zero delta ticks (instantaneous duplicate call) must not raise ZeroDivisionError."""
        monitor = DarwinIdleMonitor()
        snap = CpuTickSnapshot(user=10000, system=5000, idle=80000, nice=0, timestamp=100.0)
        monitor._last_snapshot = snap

        # Mock get_global_cpu_ticks to return the exact same snapshot (zero delta elapsed)
        with (
            mock.patch.object(monitor, "get_global_cpu_ticks", return_value=snap),
            mock.patch("time.sleep"),  # Skip real 20ms delay
        ):
            val = monitor.get_cpu_quiescence()
            # Must safely execute division guard without ZeroDivisionError and return 1.0
            assert val == 1.0
            assert isinstance(val, float)

    def test_qos_failure_returns_false_on_pthread_error(self) -> None:
        """When pthread_set_qos_class_self_np returns non-zero error, returns False."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        mock_lib = mock.MagicMock()
        # Non-zero return indicates error (e.g. 22 = EINVAL)
        mock_lib.pthread_set_qos_class_self_np.return_value = 22
        mock_lib.setiopolicy_np.return_value = 0
        monitor._libsystem = mock_lib

        result = monitor.set_background_qos()
        assert result is False
        mock_lib.pthread_set_qos_class_self_np.assert_called_once_with(QOS_CLASS_BACKGROUND, 0)

    def test_qos_failure_returns_false_on_iopolicy_error(self) -> None:
        """When setiopolicy_np returns non-zero error, returns False."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        mock_lib = mock.MagicMock()
        mock_lib.pthread_set_qos_class_self_np.return_value = 0
        mock_lib.setiopolicy_np.return_value = -1
        monitor._libsystem = mock_lib

        result = monitor.set_background_qos()
        assert result is False

    def test_qos_failure_returns_false_on_c_exception(self) -> None:
        """When low-level call raises an exception, safely handles and returns False."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        mock_lib = mock.MagicMock()
        mock_lib.pthread_set_qos_class_self_np.side_effect = OSError("Mach kernel fault")
        mock_lib.setiopolicy_np.return_value = 0
        monitor._libsystem = mock_lib

        result = monitor.set_background_qos()
        assert result is False

    def test_qos_failure_when_libsystem_none(self) -> None:
        """When _libsystem is None on Darwin, returns False safely."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        monitor._libsystem = None
        assert monitor.set_background_qos() is False

    def test_power_sources_null_defaults_to_ac(self) -> None:
        """When IOKit power source snapshot returns NULL, defaults safely to not on battery."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        mock_iokit = mock.MagicMock()
        mock_iokit.IOPSCopyPowerSourcesInfo.return_value = None
        monitor._iokit = mock_iokit
        monitor._cf = mock.MagicMock()

        # Deterministically test plugged condition (AC power)
        mock_ac = mock.MagicMock(power_plugged=True, percent=100, secsleft=-1)
        with mock.patch("psutil.sensors_battery", return_value=mock_ac):
            assert monitor.is_on_battery() is False
            telemetry = monitor.get_power_telemetry()
            assert isinstance(telemetry, PowerTelemetry)
            assert telemetry.is_ac_powered is True

        # Deterministically test on-battery condition (discharging)
        mock_bat = mock.MagicMock(power_plugged=False, percent=65, secsleft=7200)
        with mock.patch("psutil.sensors_battery", return_value=mock_bat):
            assert monitor.is_on_battery() is True
            telemetry_bat = monitor.get_power_telemetry()
            assert isinstance(telemetry_bat, PowerTelemetry)
            assert telemetry_bat.is_ac_powered is False
            assert telemetry_bat.battery_level_pct == 65

        # Deterministically test desktop fallback when battery sensor is unavailable (None)
        with mock.patch("psutil.sensors_battery", return_value=None):
            assert monitor.is_on_battery() is False
            telemetry_none = monitor.get_power_telemetry()
            assert isinstance(telemetry_none, PowerTelemetry)
            assert telemetry_none.is_ac_powered is True

    def test_power_sources_empty_list_defaults_to_ac(self) -> None:
        """When IOKit power sources list is empty (count 0), defaults safely to AC power."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        mock_iokit = mock.MagicMock()
        mock_cf = mock.MagicMock()
        mock_iokit.IOPSCopyPowerSourcesInfo.return_value = 0x1234
        mock_iokit.IOPSCopyPowerSourcesList.return_value = 0x5678
        mock_cf.CFArrayGetCount.return_value = 0
        mock_iokit.IOPSGetProvidingPowerSourceType.return_value = None
        monitor._iokit = mock_iokit
        monitor._cf = mock_cf

        telemetry = monitor.get_power_telemetry()
        assert telemetry.is_ac_powered is True
        assert telemetry.battery_health == "None"
        assert monitor.is_on_battery() is False

    def test_thermal_state_unsupported_defaults_nominal(self) -> None:
        """When both ObjC and notify(3) queries fail, safely defaults to Nominal (0)."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        with (
            mock.patch.object(monitor, "_get_thermal_state_objc", return_value=None),
            mock.patch.object(monitor, "_get_thermal_state_notify", return_value=None),
        ):
            assert monitor.get_thermal_state() == 0

    def test_thermal_state_notify_error_handling(self) -> None:
        """When notify_get_state returns non-zero error, _get_thermal_state_notify returns None."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True
        mock_lib = mock.MagicMock()
        mock_lib.notify_get_state.return_value = -1
        monitor._libsystem = mock_lib
        monitor._thermal_token = ctypes.c_int(99)

        assert monitor._get_thermal_state_notify() is None

    @pytest.mark.parametrize(
        ("raw_state", "expected_thermal"),
        [
            (0, 0),   # Nominal
            (1, 1),   # Fair
            (10, 1),  # Fair (alternative code)
            (20, 1),  # Fair (alternative code)
            (2, 2),   # Serious
            (30, 2),  # Serious (alternative code)
            (3, 3),   # Critical
            (4, 3),   # Critical
        ],
    )
    def test_thermal_state_notify_raw_mappings(self, raw_state: int, expected_thermal: int) -> None:
        """Verifies notify(3) raw states correctly map to Darwin thermal states 0..3."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = True

        def fake_notify_get_state(token: int, state_ref: ctypes.c_void_p) -> int:
            ctypes.cast(state_ref, ctypes.POINTER(ctypes.c_uint64)).contents.value = raw_state
            return 0

        mock_lib = mock.MagicMock()
        mock_lib.notify_get_state.side_effect = fake_notify_get_state
        monitor._libsystem = mock_lib
        monitor._thermal_token = ctypes.c_int(42)

        assert monitor._get_thermal_state_notify() == expected_thermal


# ============================================================================
# Tier 1: Cross-Platform Fallbacks (No Facade Mocks)
# ============================================================================


class TestCrossPlatformFallbacks:
    """Verifies that non-Darwin platforms (Linux, Windows) fail-soft gracefully.

    Mocks platform indicators and OS-level primitives (os.setpriority, os.nice,
    ctypes.windll) to exercise genuine fallback branches.
    """

    def test_linux_platform_fallback_setpriority(self) -> None:
        """Simulates Linux environment where os.setpriority succeeds."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False
        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", True),
            mock.patch("quanta.cognitive.darwin_idle.IS_DARWIN", False),
            mock.patch("quanta.cognitive.darwin_idle.IS_WINDOWS", False),
            mock.patch("os.setpriority") as mock_setpriority,
        ):
            assert monitor.set_background_qos() is True
            mock_setpriority.assert_called_once_with(os.PRIO_PROCESS, 0, 19)

    def test_linux_platform_fallback_os_nice(self) -> None:
        """Simulates Linux where os.setpriority fails and os.nice(19) succeeds."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False
        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", True),
            mock.patch("quanta.cognitive.darwin_idle.IS_DARWIN", False),
            mock.patch("quanta.cognitive.darwin_idle.IS_WINDOWS", False),
            mock.patch("os.setpriority", side_effect=PermissionError),
            mock.patch("os.nice") as mock_nice,
        ):
            assert monitor.set_background_qos() is True
            mock_nice.assert_called_once_with(19)

    def test_linux_platform_fallback_all_fail(self) -> None:
        """Simulates Linux where both os.setpriority and os.nice fail."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False
        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", True),
            mock.patch("quanta.cognitive.darwin_idle.IS_DARWIN", False),
            mock.patch("quanta.cognitive.darwin_idle.IS_WINDOWS", False),
            mock.patch("os.setpriority", side_effect=PermissionError),
            mock.patch("os.nice", side_effect=OSError),
        ):
            assert monitor.set_background_qos() is False

    def test_windows_platform_fallback_success(self) -> None:
        """Simulates Windows environment with SetPriorityClass and SetThreadPriority."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False
        mock_k32 = mock.MagicMock()
        mock_k32.GetCurrentProcess.return_value = 0x1000
        mock_k32.GetCurrentThread.return_value = 0x2000
        mock_k32.SetPriorityClass.return_value = 1
        mock_k32.SetThreadPriority.return_value = 1
        mock_windll = mock.MagicMock(kernel32=mock_k32)

        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_WINDOWS", True),
            mock.patch("quanta.cognitive.darwin_idle.IS_DARWIN", False),
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", False),
            mock.patch("ctypes.windll", mock_windll, create=True),
        ):
            assert monitor.set_background_qos() is True
            mock_k32.SetPriorityClass.assert_called_once_with(0x1000, 0x00000040)
            mock_k32.SetThreadPriority.assert_called_once_with(0x2000, -15)

    def test_windows_platform_fallback_failure(self) -> None:
        """Simulates Windows environment where SetPriorityClass fails."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False
        mock_k32 = mock.MagicMock()
        mock_k32.GetCurrentProcess.return_value = 0x1000
        mock_k32.SetPriorityClass.side_effect = OSError("Access denied")
        mock_windll = mock.MagicMock(kernel32=mock_k32)

        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_WINDOWS", True),
            mock.patch("quanta.cognitive.darwin_idle.IS_DARWIN", False),
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", False),
            mock.patch("ctypes.windll", mock_windll, create=True),
        ):
            assert monitor.set_background_qos() is False

    def test_unsupported_platform_fallback(self) -> None:
        """On an unsupported operating system, set_background_qos safely returns False."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False
        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_WINDOWS", False),
            mock.patch("quanta.cognitive.darwin_idle.IS_DARWIN", False),
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", False),
        ):
            assert monitor.set_background_qos() is False

    def test_linux_proc_stat_cpu_ticks_parsing(self) -> None:
        """Verifies parsing of Linux /proc/stat CPU tick counters into CpuTickSnapshot."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False

        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", True),
            mock.patch("quanta.cognitive.darwin_idle.IS_DARWIN", False),
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch(
                "builtins.open",
                mock.mock_open(read_data="cpu  12345 67 8901 234567 0 0 0 0 0 0\n"),
            ),
        ):
            snap = monitor.get_global_cpu_ticks()
            assert snap.user == 12345
            assert snap.nice == 67
            assert snap.system == 8901
            assert snap.idle == 234567

    @pytest.mark.parametrize(
        ("temp_mc", "expected_thermal"),
        [
            (50000, 0),  # 50 C -> Nominal
            (68000, 1),  # 68 C -> Fair (>=65)
            (82000, 2),  # 82 C -> Serious (>=80)
            (94000, 3),  # 94 C -> Critical (>=90)
        ],
    )
    def test_linux_thermal_zone_parsing(self, temp_mc: int, expected_thermal: int) -> None:
        """Verifies parsing of Linux /sys/class/thermal zones into thermal states."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False

        mock_zone = mock.MagicMock()
        mock_zone.read_text.return_value = str(temp_mc)

        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", True),
            mock.patch("quanta.cognitive.darwin_idle.Path.exists", return_value=True),
            mock.patch("quanta.cognitive.darwin_idle.Path.glob", return_value=[mock_zone]),
        ):
            assert monitor._get_thermal_state_linux() == expected_thermal

    def test_linux_per_core_ticks_parsing(self) -> None:
        """Verifies parsing of Linux /proc/stat per-core ticks."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False

        proc_data = (
            "cpu  1000 200 300 4000\n"
            "cpu0 500 100 150 2000\n"
            "cpu1 500 100 150 2000\n"
        )
        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", True),
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch("builtins.open", mock.mock_open(read_data=proc_data)),
        ):
            snapshots = monitor.get_per_core_cpu_ticks()
            assert len(snapshots) == 2
            assert snapshots[0].user == 500
            assert snapshots[1].idle == 2000

    def test_linux_power_supply_parsing(self) -> None:
        """Verifies parsing of Linux /sys/class/power_supply."""
        monitor = DarwinIdleMonitor()
        mock_ac = mock.MagicMock()
        mock_ac.read_text.return_value = "1"
        mock_bat = mock.MagicMock()
        mock_bat.read_text.return_value = "85"

        def glob_side_effect(pat: str) -> list[Any]:
            if "AC*" in pat:
                return [mock_ac]
            if "BAT*" in pat:
                return [mock_bat]
            return []

        with (
            mock.patch.dict("sys.modules", {"psutil": None}),
            mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", True),
            mock.patch("quanta.cognitive.darwin_idle.Path.exists", return_value=True),
            mock.patch("quanta.cognitive.darwin_idle.Path.glob", side_effect=glob_side_effect),
        ):
            pwr = monitor._get_fallback_power_telemetry()
            assert pwr.is_ac_powered is True
            assert pwr.battery_level_pct == 85

    def test_windows_get_system_times_fallback(self) -> None:
        """Verifies parsing of Windows GetSystemTimes ticks."""
        monitor = DarwinIdleMonitor()
        monitor.is_darwin = False

        mock_k32 = mock.MagicMock()
        def mock_get_system_times(idle_ptr: Any, kern_ptr: Any, user_ptr: Any) -> bool:
            idle_ptr._obj.dwLowDateTime = 1000
            idle_ptr._obj.dwHighDateTime = 0
            kern_ptr._obj.dwLowDateTime = 2500
            kern_ptr._obj.dwHighDateTime = 0
            user_ptr._obj.dwLowDateTime = 1500
            user_ptr._obj.dwHighDateTime = 0
            return True
        mock_k32.GetSystemTimes.side_effect = mock_get_system_times

        mock_windll = mock.MagicMock()
        mock_windll.kernel32 = mock_k32

        with (
            mock.patch("quanta.cognitive.darwin_idle.IS_DARWIN", False),
            mock.patch("quanta.cognitive.darwin_idle.IS_WINDOWS", True),
            mock.patch.object(ctypes, "windll", mock_windll, create=True),
        ):
            snap = monitor.get_global_cpu_ticks()
            assert snap.idle == 1000
            assert snap.user == 1500
            assert snap.system == 1500  # kern(2500) - idle(1000)



# ============================================================================
# Tier 1: 32-Bit Mach Tick Counter Rollover ($2^{32}-1 \to 0$)
# ============================================================================


class TestDarwin32BitCounterOverflow:
    """Verifies that 32-bit unsigned tick counter rollovers do not break quiescence.

    In the Darwin XNU Mach kernel, HostCpuLoadInfo counters are 32-bit unsigned
    integers (c_uint, max 4,294,967,295). Modular arithmetic (& 0xFFFFFFFF) must
    correctly recover positive tick deltas and prevent false 1.0 (100% idle) reports.
    """

    def test_user_counter_rollover_under_heavy_cpu_load(self) -> None:
        """Rollover on user ticks during heavy load must NOT report 100% idle (1.0).

        Scenario: User counter wraps around from 4,294,967,000 to 500 across 2^32-1.
        Delta user: (500 - 4294967000) & 0xFFFFFFFF = 796 ticks.
        Delta system: 50 ticks.
        Delta idle: 100 ticks.
        Delta total: 796 + 50 + 100 = 946 ticks.
        Expected idle ratio: 100 / 946 ≈ 0.1057 (approx 10.57% idle, high load).
        """
        monitor = DarwinIdleMonitor()
        prev = CpuTickSnapshot(user=4294967000, system=1000, idle=1000, nice=0, timestamp=100.0)
        curr = CpuTickSnapshot(user=500, system=1050, idle=1100, nice=0, timestamp=101.0)
        monitor._last_snapshot = prev

        with mock.patch.object(monitor, "get_global_cpu_ticks", return_value=curr):
            quiescence = monitor.get_cpu_quiescence()

            expected_ratio = 100 / (796 + 50 + 100)
            assert quiescence == pytest.approx(expected_ratio, abs=1e-4)
            # The system is under heavy load; must NOT report 1.0
            assert quiescence < 0.20
            assert quiescence != 1.0

    def test_idle_counter_rollover_under_idle_system(self) -> None:
        """Rollover on idle ticks must correctly compute high idle ratio.

        Scenario: Idle counter wraps around from 4,294,967,290 to 10 across 2^32-1.
        Delta idle: (10 - 4294967290) & 0xFFFFFFFF = 16 ticks.
        Delta user: 5 ticks.
        Delta system: 5 ticks.
        Delta total: 5 + 5 + 16 = 26 ticks.
        Expected idle ratio: 16 / 26 ≈ 0.6154.
        """
        monitor = DarwinIdleMonitor()
        prev = CpuTickSnapshot(user=100, system=100, idle=4294967290, nice=0, timestamp=100.0)
        curr = CpuTickSnapshot(user=105, system=105, idle=10, nice=0, timestamp=101.0)
        monitor._last_snapshot = prev

        with mock.patch.object(monitor, "get_global_cpu_ticks", return_value=curr):
            quiescence = monitor.get_cpu_quiescence()
            expected_ratio = 16 / 26
            assert quiescence == pytest.approx(expected_ratio, abs=1e-4)

    def test_exact_max_boundary_rollover_0xffffffff_to_0x10(self) -> None:
        """Rollover across exact 0xFFFFFFFF (2^32-1) boundary produces positive delta.

        From 0xFFFFFFFF (4294967295) to 0x00000010 (16) is exactly 17 ticks:
        (16 - 4294967295) & 0xFFFFFFFF == 17.
        Idle delta: 83 ticks. Total: 100 ticks.
        Expected idle ratio: 83 / 100 = 0.83.
        """
        monitor = DarwinIdleMonitor()
        prev = CpuTickSnapshot(user=0xFFFFFFFF, system=1000, idle=1000, nice=0, timestamp=100.0)
        curr = CpuTickSnapshot(user=0x00000010, system=1000, idle=1083, nice=0, timestamp=101.0)
        monitor._last_snapshot = prev

        with mock.patch.object(monitor, "get_global_cpu_ticks", return_value=curr):
            quiescence = monitor.get_cpu_quiescence()
            assert quiescence == pytest.approx(0.83, abs=1e-4)

    def test_all_counters_concurrent_rollover(self) -> None:
        """All 4 core state counters rolling over simultaneously yields valid ratio in [0, 1].

        user: 0xFFFFFFF0 -> 0x00000010 (32 ticks)
        system: 0xFFFFFFF0 -> 0x00000010 (32 ticks)
        idle: 0xFFFFFFF0 -> 0x00000060 (112 ticks)
        nice: 0 -> 0 (0 ticks)
        total: 32 + 32 + 112 = 176 ticks.
        Expected idle ratio: 112 / 176 ≈ 0.6364.
        """
        monitor = DarwinIdleMonitor()
        prev = CpuTickSnapshot(
            user=0xFFFFFFF0, system=0xFFFFFFF0, idle=0xFFFFFFF0, nice=0, timestamp=100.0
        )
        curr = CpuTickSnapshot(
            user=0x00000010, system=0x00000010, idle=0x00000060, nice=0, timestamp=101.0
        )
        monitor._last_snapshot = prev

        with mock.patch.object(monitor, "get_global_cpu_ticks", return_value=curr):
            quiescence = monitor.get_cpu_quiescence()
            expected_ratio = 112 / 176
            assert quiescence == pytest.approx(expected_ratio, abs=1e-4)
            assert 0.0 <= quiescence <= 1.0

    def test_compute_cpu_utilization_across_rollover(self) -> None:
        """compute_cpu_utilization returns valid (load_pct, idle_pct) summing to 100%
        across 32-bit counter rollover.
        """
        monitor = DarwinIdleMonitor()
        prev = CpuTickSnapshot(user=4294967000, system=1000, idle=1000, nice=0, timestamp=100.0)
        curr = CpuTickSnapshot(user=500, system=1050, idle=1100, nice=0, timestamp=101.0)
        monitor._last_snapshot = prev

        with mock.patch.object(monitor, "get_global_cpu_ticks", return_value=curr):
            load_pct, idle_pct = monitor.compute_cpu_utilization()
            assert abs((load_pct + idle_pct) - 100.0) <= 0.05
            expected_idle = round((100 / (796 + 50 + 100)) * 100.0, 2)
            assert idle_pct == pytest.approx(expected_idle, abs=0.1)


# ============================================================================
# Tier 1: Thread Safety & Concurrency Synchronization
# ============================================================================


class TestDarwinThreadSafety:
    """Verifies that DarwinIdleMonitor and get_default_monitor are thread-safe."""

    def test_monitor_has_threading_lock(self) -> None:
        """DarwinIdleMonitor instance must contain a threading.Lock instance."""
        monitor = DarwinIdleMonitor()
        assert hasattr(monitor, "_lock")
        assert isinstance(monitor._lock, type(threading.Lock()))

    def test_concurrent_quiescence_access_serialized(self) -> None:
        """8 concurrent worker threads querying get_cpu_quiescence() run without race conditions."""
        monitor = DarwinIdleMonitor()
        errors: list[tuple[int, Exception]] = []

        def worker(wid: int) -> None:
            try:
                for _ in range(10):
                    q = monitor.get_cpu_quiescence()
                    assert 0.0 <= q <= 1.0
            except Exception as e:
                errors.append((wid, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_default_monitor_singleton_thread_safe(self) -> None:
        """Concurrent calls to get_default_monitor() return the exact same singleton instance."""
        instances: list[DarwinIdleMonitor] = []
        lock = threading.Lock()

        def worker() -> None:
            inst = get_default_monitor()
            with lock:
                instances.append(inst)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 16
        first = instances[0]
        for inst in instances:
            assert inst is first


# ============================================================================
# Tier 1: Pairwise Combinatorial Test Frames
# ============================================================================


class TestPairwiseHardwareFrames:
    """Pairwise combinatorial test frames for hardware and quiescence states."""

    def test_frame_tf01_darwin_nominal_ac(self) -> None:
        """TF-01: Darwin, High Idle (92%), Nominal Thermal (0), AC Power -> Idle=True."""
        with (
            mock.patch.object(DarwinIdleMonitor, "get_cpu_quiescence", return_value=0.92),
            mock.patch.object(DarwinIdleMonitor, "get_thermal_state", return_value=0),
            mock.patch.object(DarwinIdleMonitor, "is_on_battery", return_value=False),
        ):
            assert is_system_idle() is True
            assert is_on_battery() is False

    def test_frame_tf03_darwin_busy_cpu(self) -> None:
        """TF-03: Darwin, Low Idle (60%), Nominal Thermal (0), AC Power -> Idle=False."""
        with (
            mock.patch.object(DarwinIdleMonitor, "get_cpu_quiescence", return_value=0.60),
            mock.patch.object(DarwinIdleMonitor, "get_thermal_state", return_value=0),
        ):
            assert is_system_idle() is False

    def test_frame_tf04_darwin_thermal_throttling(self) -> None:
        """TF-04: Darwin, High Idle (95%), Serious Thermal (2), AC Power -> Idle=False."""
        with (
            mock.patch.object(DarwinIdleMonitor, "get_cpu_quiescence", return_value=0.95),
            mock.patch.object(DarwinIdleMonitor, "get_thermal_state", return_value=2),
        ):
            assert is_system_idle() is False

    def test_frame_tf07_darwin_fair_thermal(self) -> None:
        """TF-07: Darwin, High Idle (91%), Fair Thermal (1), AC Power -> Idle=True."""
        with (
            mock.patch.object(DarwinIdleMonitor, "get_cpu_quiescence", return_value=0.91),
            mock.patch.object(DarwinIdleMonitor, "get_thermal_state", return_value=1),
        ):
            assert is_system_idle(max_thermal=1) is True

    def test_darwin_per_core_ticks_and_telemetry(self) -> None:
        """Covers get_per_core_cpu_ticks, assess_quiescence, and get_power_telemetry."""
        monitor = get_default_monitor()
        per_core = monitor.get_per_core_cpu_ticks()
        assert isinstance(per_core, list)

        state = monitor.assess_quiescence()
        assert isinstance(state, HardwareQuiescenceState)
        assert isinstance(state.is_quiescent, bool)
        assert 0.0 <= state.cpu_idle_pct <= 100.0

        pwr = monitor.get_power_telemetry()
        assert isinstance(pwr, PowerTelemetry)

        fallback_pwr = monitor._get_fallback_power_telemetry()
        assert isinstance(fallback_pwr, PowerTelemetry)
        assert isinstance(fallback_pwr.is_ac_powered, bool)

        # Deterministically test fallback logic under both AC and battery mocks
        mock_ac = mock.MagicMock(power_plugged=True, percent=95, secsleft=-1)
        with mock.patch("psutil.sensors_battery", return_value=mock_ac):
            pwr_ac = monitor._get_fallback_power_telemetry()
            assert pwr_ac.is_ac_powered is True
            assert pwr_ac.battery_level_pct == 95

        mock_bat = mock.MagicMock(power_plugged=False, percent=40, secsleft=3600)
        with mock.patch("psutil.sensors_battery", return_value=mock_bat):
            pwr_bat = monitor._get_fallback_power_telemetry()
            assert pwr_bat.is_ac_powered is False
            assert pwr_bat.battery_level_pct == 40

    def test_snapshot_methods_and_darwin_coverage_gaps(self) -> None:
        """Covers CpuTickSnapshot methods and edge branches."""
        from quanta.cognitive.darwin_idle import CpuTickSnapshot

        snap = CpuTickSnapshot(user=10, system=20, idle=70, nice=0, timestamp=100.0)
        assert snap.total == 100
        assert len(snap) == 4
        assert list(snap) == [10, 20, 70, 0]
        assert snap[0] == 10
        assert snap[1] == 20
        assert snap[2] == 70
        assert snap[3] == 0

        with pytest.raises(IndexError):
            _ = snap[4]

    def test_darwin_active_qos_and_thermal_branches(self) -> None:
        """Covers get_active_qos, ObjC and notify thermal states, and thermal pressure."""
        monitor = get_default_monitor()
        qos, prio = monitor.get_active_qos()
        assert isinstance(qos, int)
        assert isinstance(prio, int)

        t_objc = monitor._get_thermal_state_objc()
        assert t_objc is None or 0 <= t_objc <= 3

        t_notify = monitor._get_thermal_state_notify()
        assert t_notify is None or 0 <= t_notify <= 3

        t_pressure = monitor.get_thermal_pressure()
        assert 0 <= t_pressure <= 3

        # Test set_background_qos
        res = monitor.set_background_qos()
        assert isinstance(res, bool)

        # Mock non-darwin branch for get_active_qos
        with mock.patch.object(monitor, "is_darwin", False):
            assert monitor.get_active_qos() == (0, 0)
            assert monitor._get_thermal_state_objc() is None
            assert monitor._get_thermal_state_notify() is None

    def test_darwin_additional_coverage_paths(self, tmp_path: Path) -> None:
        """Covers prev is None branch, linux thermal zones, and linux power supply fallback."""
        monitor = DarwinIdleMonitor()
        monitor._last_snapshot = None
        q = monitor.get_cpu_quiescence()
        assert 0.0 <= q <= 1.0

        # Mock Linux thermal zones
        thermal_dir = tmp_path / "sys" / "class" / "thermal"
        thermal_dir.mkdir(parents=True)
        zone1 = thermal_dir / "thermal_zone0"
        zone1.mkdir()
        (zone1 / "temp").write_text("95000")  # 95C -> Critical

        with mock.patch("quanta.cognitive.darwin_idle.IS_LINUX", True), \
             mock.patch("quanta.cognitive.darwin_idle.Path", return_value=thermal_dir):
            t_state = monitor._get_thermal_state_linux()
            assert t_state == 3  # Critical




