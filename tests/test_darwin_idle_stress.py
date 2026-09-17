"""Empirical stress tests and adversarial verification for Darwin Mach kernel QoS.

Written by teamwork_preview_challenger_m1_1.
Tests quanta.cognitive.darwin_idle directly against hardware without mocks:
1. Rapid back-to-back get_cpu_quiescence() (100 iterations) verifying stability and deltas.
2. set_background_qos() and pthread_get_qos_class_np verification of active class 0x09.
3. Repeated thermal queries (1000 iterations) without crashing.
4. Repeated battery and power telemetry queries (1000 iterations) verifying no memory bloat.
5. Multi-threaded concurrency stress and adversarial boundary conditions.
"""

from __future__ import annotations

import gc
import platform
import resource
import threading
import time

import pytest

from quanta.cognitive.darwin_idle import (
    QOS_CLASS_BACKGROUND,
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

DARWIN_SKIP = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Empirical tests require macOS Darwin host",
)


@DARWIN_SKIP
class TestDarwinCpuQuiescenceStress:
    """Empirical challenge of get_cpu_quiescence() under rapid back-to-back execution."""

    def test_rapid_back_to_back_quiescence_100_iterations(self) -> None:
        """Requirement 2: 100 iterations to verify stability and delta calculations."""
        results: list[float] = []
        t0 = time.perf_counter()

        for _ in range(100):
            val = get_cpu_quiescence()
            results.append(val)

        t1 = time.perf_counter()
        elapsed = t1 - t0

        assert len(results) == 100
        # Verify every returned ratio is a float in [0.0, 1.0]
        for idx, val in enumerate(results):
            assert isinstance(val, float), f"Iteration {idx} non-float: {type(val)}"
            assert 0.0 <= val <= 1.0, f"Iteration {idx} out-of-range ratio: {val}"

        # Average duration reflects the 20ms micro-sleep when d_total <= 0
        avg_ms = (elapsed / 100) * 1000
        assert avg_ms >= 15.0, f"Average duration ({avg_ms:.2f}ms) too low; sleep skipped?"

    def test_mach_tick_monotonicity_and_delta_invariants(self) -> None:
        """Verify Mach tick counters remain strictly non-decreasing between snapshots."""
        monitor = DarwinIdleMonitor()
        prev_snap = monitor.get_global_cpu_ticks()

        for _ in range(30):
            time.sleep(0.01)
            curr_snap = monitor.get_global_cpu_ticks()

            assert curr_snap.user >= prev_snap.user, "User ticks decreased"
            assert curr_snap.system >= prev_snap.system, "System ticks decreased"
            assert curr_snap.idle >= prev_snap.idle, "Idle ticks decreased"
            assert curr_snap.nice >= prev_snap.nice, "Nice ticks decreased"

            total_delta = curr_snap.total - prev_snap.total
            assert total_delta >= 0, "Total ticks decreased"
            prev_snap = curr_snap


@DARWIN_SKIP
class TestDarwinPthreadQoSStress:
    """Empirical challenge of set_background_qos() and pthread_get_qos_class_np (0x09)."""

    def test_set_background_qos_sets_active_class_to_0x09(self) -> None:
        """Requirement 3: Test set_background_qos() and query QoS via pthread_get_qos_class_np."""
        monitor = DarwinIdleMonitor()
        assert monitor._libsystem is not None, "libSystem bindings not initialized"

        success = set_background_qos()
        assert success is True, "set_background_qos() returned False on Darwin host"

        active_qos, relative_prio = monitor.get_active_qos()
        assert active_qos == QOS_CLASS_BACKGROUND, (
            f"Active QoS class is {hex(active_qos)}, expected {hex(QOS_CLASS_BACKGROUND)}"
        )
        assert active_qos == 0x09, f"Expected numeric 0x09, got {hex(active_qos)}"
        assert relative_prio == 0, f"Expected relative priority 0, got {relative_prio}"

    def test_worker_threads_can_independently_set_background_qos_0x09(self) -> None:
        """Verify new threads independently achieve QOS_CLASS_BACKGROUND (0x09)."""
        monitor = DarwinIdleMonitor()
        results: dict[str, int] = {}

        def thread_target() -> None:
            set_background_qos()
            qos, _ = monitor.get_active_qos()
            results["thread_qos"] = qos

        t = threading.Thread(target=thread_target)
        t.start()
        t.join()

        assert results.get("thread_qos") == 0x09, (
            f"Worker thread failed to achieve QoS 0x09; got {results.get('thread_qos')}"
        )


@DARWIN_SKIP
class TestDarwinThermalAndBatteryStress:
    """Empirical challenge of thermal and battery queries under repeated execution."""

    def test_repeated_thermal_queries_stability(self) -> None:
        """Requirement 4 (Thermal): 1000 repeated calls without crashing."""
        monitor = get_default_monitor()
        states_seen: set[int] = set()

        t0 = time.perf_counter()
        for _ in range(1000):
            th = monitor.get_thermal_state()
            states_seen.add(th)
        t1 = time.perf_counter()

        # All observed thermal states must be within valid Darwin NSProcessInfo states (0..3)
        assert states_seen.issubset({0, 1, 2, 3}), f"Unexpected thermal state: {states_seen}"
        avg_us = ((t1 - t0) / 1000) * 1_000_000
        assert avg_us < 100.0, f"Thermal query latency too high: {avg_us:.2f} µs/call"

    def test_repeated_battery_queries_and_memory_bloat(self) -> None:
        """Requirement 4 (Battery & Memory): 1000 repeated calls with < 10 MB growth."""
        monitor = get_default_monitor()

        def get_rss_bytes() -> int:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        gc.collect()
        rss_start = get_rss_bytes()

        last_telemetry: PowerTelemetry | None = None
        for _ in range(1000):
            last_telemetry = monitor.get_power_telemetry()
            on_battery = monitor.is_on_battery()
            assert isinstance(on_battery, bool)

        gc.collect()
        rss_end = get_rss_bytes()
        rss_delta_mb = (rss_end - rss_start) / (1024 * 1024)

        assert last_telemetry is not None
        assert isinstance(last_telemetry.is_ac_powered, bool)
        batt = last_telemetry.battery_level_pct
        assert batt is None or (0 <= batt <= 100)

        # Confirm CoreFoundation objects are properly released: RSS growth must be < 10 MB
        assert rss_delta_mb < 10.0, f"Potential memory leak: RSS grew by {rss_delta_mb:.2f} MB"


class TestAdversarialAndConcurrentStress:
    """Stress testing edge cases, boundary parameters, and concurrent threads."""

    def test_assess_quiescence_invariants(self) -> None:
        """Verify HardwareQuiescenceState invariants across various thresholds."""
        monitor = get_default_monitor()
        state = monitor.assess_quiescence(min_idle_pct=70.0, max_thermal=1)

        assert isinstance(state, HardwareQuiescenceState)
        assert 0.0 <= state.cpu_load_pct <= 100.0
        assert 0.0 <= state.cpu_idle_pct <= 100.0
        # Load + Idle sum must equal 100.0 (+/- 0.05 rounding margin)
        assert abs((state.cpu_load_pct + state.cpu_idle_pct) - 100.0) <= 0.05
        assert state.thermal_pressure_level in (0, 1, 2, 3)
        assert state.power_source in ("AC Power", "Battery Power")

    def test_boundary_parameter_safety(self) -> None:
        """Extreme/invalid boundary parameters must not raise exceptions."""
        assert is_system_idle(idle_threshold=-1.0) in (True, False)
        assert is_system_idle(idle_threshold=2.0) is False
        assert is_system_idle(max_thermal=-1) is False
        assert is_system_idle(max_thermal=10) in (True, False)

    def test_multi_threaded_concurrency_stress(self) -> None:
        """8 concurrent threads querying telemetry simultaneously without deadlock/crash."""
        monitor = get_default_monitor()
        errors: list[tuple[int, Exception]] = []

        def worker(wid: int) -> None:
            try:
                for _ in range(15):
                    q = get_cpu_quiescence()
                    th = get_thermal_state()
                    bat = is_on_battery()
                    st = monitor.assess_quiescence()
                    assert 0.0 <= q <= 1.0
                    assert th in (0, 1, 2, 3)
                    assert isinstance(bat, bool)
                    assert 0.0 <= st.cpu_idle_pct <= 100.0
            except Exception as e:
                errors.append((wid, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent workers encountered errors: {errors}"
