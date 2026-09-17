"""Empirical adversarial stress test suite for PoissonSpindleTrigger.

Empirical verification covering:
1. Kolmogorov-Smirnov Goodness-of-Fit via Time-Rescaling Theorem under varying
   dynamic rates lambda(t) for n=1,000 renewal intervals (Lewis-Shedler thinning,
   discrete epoch should_trigger, and multi-seed Monte Carlo resilience).
2. Absolute refractory dead-time boundary stress: 100,000 calls within t < t_last + 10.0,
   boundary sub-microsecond precision, clock jitter / negative delta handling.
3. Extreme rate conditions: rate = 0.0, rate = 10^6, negative rate inputs, NaN/Inf handling,
   sigmoid saturation bounds, and numerical stability.
4. Property-based fuzzing via Hypothesis.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from quanta.cognitive.poisson_trigger import PoissonSpindleTrigger

# ============================================================================
# 1. Kolmogorov-Smirnov Goodness-of-Fit via Time-Rescaling Theorem (n=1,000)
# ============================================================================


class TestTimeRescalingKolmogorovSmirnovStress:
    """Stress tests statistical point process renewal adherence with n=1,000 intervals."""

    def test_ks_time_rescaling_thinning_dynamic_rates_1000(self) -> None:
        """Verify Time-Rescaling Theorem on n=1,000 intervals under dynamic rate lambda(t).

        Uses Lewis-Shedler thinning algorithm to draw exact arrivals from the inhomogeneous
        Poisson point process whose hazard rate modulates dynamically via user activity,
        fatigue oscillations, and Theory of Mind urgency shifts.
        """
        random.seed(42)
        np.random.seed(42)

        trigger = PoissonSpindleTrigger(
            lambda_0=0.2, idle_min=15.0, tau=5.0, refractory_sec=0.0
        )

        def dynamic_rate_fn(t: float) -> float:
            cycle = t % 120.0
            last_active = t - cycle if cycle > 15.0 else t - cycle - 120.0
            fatigue = 0.25 + 0.25 * math.sin(2.0 * math.pi * t / 400.0)
            urgency = 1.2 + 0.6 * math.cos(2.0 * math.pi * t / 250.0)
            return trigger.compute_rate(
                t, last_active, fatigue=fatigue, tom_urgency=urgency
            )

        lambda_sup = 0.2 * 1.0 * 1.0 * (1.2 + 0.6)  # 0.36 Hz
        n_events = 1000
        event_times: list[float] = []
        t = 0.0

        while len(event_times) < n_events:
            u = random.random()
            dt = -math.log(max(1e-12, u)) / lambda_sup
            t += dt
            rate = dynamic_rate_fn(t)
            if random.random() < (rate / lambda_sup):
                event_times.append(t)

        # Time-Rescaling Theorem: Lambda_k = integral_{t_{k-1}}^{t_k} lambda(s) ds
        rescaled_intervals: list[float] = []
        t_prev = 0.0
        steps = 50
        for t_curr in event_times:
            ds = (t_curr - t_prev) / steps
            integral = sum(
                dynamic_rate_fn(t_prev + (i + 0.5) * ds) * ds for i in range(steps)
            )
            rescaled_intervals.append(integral)
            t_prev = t_curr

        # Probability Integral Transform: u_k = 1 - exp(-Lambda_k) ~ Uniform(0, 1)
        u_coords = [1.0 - math.exp(-lk) for lk in rescaled_intervals]
        ks_result = stats.kstest(u_coords, "uniform")
        d_crit = 1.36 / math.sqrt(n_events)

        assert (
            ks_result.statistic < d_crit
        ), f"KS statistic D={ks_result.statistic:.5f} exceeds D_crit={d_crit:.5f}"
        assert (
            ks_result.pvalue > 0.01
        ), f"KS test p-value {ks_result.pvalue:.5f} <= 0.01 (rejected null hypothesis)"

    def test_ks_time_rescaling_discrete_epoch_1000(self) -> None:
        """Verify Time-Rescaling on n=1,000 intervals generated via discrete should_trigger."""
        random.seed(1234)
        np.random.seed(1234)

        dt_tick = 0.05
        trigger = PoissonSpindleTrigger(
            lambda_0=0.25,
            idle_min=15.0,
            tau=5.0,
            refractory_sec=0.0,
            dt_tick=dt_tick,
        )
        t = 0.0
        last_dream = -100.0
        rescaled_intervals: list[float] = []
        current_integral = 0.0
        n_events = 1000

        while len(rescaled_intervals) < n_events:
            cycle = t % 150.0
            last_active = t - cycle if cycle > 15.0 else t - cycle - 150.0
            fatigue = 0.2 + 0.2 * math.sin(2.0 * math.pi * t / 500.0)
            urgency = 1.0 + 0.4 * math.cos(2.0 * math.pi * t / 350.0)
            rate = trigger.compute_rate(
                t, last_active, fatigue=fatigue, tom_urgency=urgency
            )
            current_integral += rate * dt_tick

            triggered = trigger.should_trigger(
                current_time=t,
                last_active_time=last_active,
                last_dream_time=last_dream,
                fatigue=fatigue,
                tom_urgency=urgency,
                dt_tick=dt_tick,
            )
            if triggered:
                rescaled_intervals.append(current_integral)
                current_integral = 0.0
                last_dream = t
            t += dt_tick

        u_coords = [1.0 - math.exp(-lk) for lk in rescaled_intervals]
        ks_result = stats.kstest(u_coords, "uniform")
        d_crit = 1.36 / math.sqrt(n_events)

        assert (
            ks_result.statistic < d_crit
        ), f"Discrete epoch KS D={ks_result.statistic:.5f} >= D_crit={d_crit:.5f}"
        assert (
            ks_result.pvalue > 0.01
        ), f"Discrete epoch KS p={ks_result.pvalue:.5f} <= 0.01"

    def test_ks_multi_seed_monte_carlo_resilience(self) -> None:
        """Run 15 distinct seeds to assert no structural statistical bias in renewal generator."""
        trigger = PoissonSpindleTrigger(
            lambda_0=0.2, idle_min=10.0, tau=5.0, refractory_sec=0.0
        )
        n_events = 1000
        d_crit = 1.36 / math.sqrt(n_events)
        pass_count = 0
        p_values: list[float] = []

        def rate_eval(t: float) -> float:
            fat = 0.15 + 0.15 * math.cos(2 * math.pi * t / 600.0)
            urg = 1.1 + 0.3 * math.sin(2 * math.pi * t / 300.0)
            return trigger.compute_rate(t, 0.0, fatigue=fat, tom_urgency=urg)

        lambda_sup = 0.2 * 1.0 * 1.0 * 1.4

        for seed in range(15):
            random.seed(seed + 100)
            event_times: list[float] = []
            t = 0.0
            while len(event_times) < n_events:
                u = random.random()
                t += -math.log(max(1e-12, u)) / lambda_sup
                r = rate_eval(t)
                if random.random() < (r / lambda_sup):
                    event_times.append(t)

            rescaled: list[float] = []
            t_prev = 0.0
            steps = 30
            for t_curr in event_times:
                ds = (t_curr - t_prev) / steps
                integral = sum(
                    rate_eval(t_prev + (i + 0.5) * ds) * ds for i in range(steps)
                )
                rescaled.append(integral)
                t_prev = t_curr

            u_coords = [1.0 - math.exp(-lk) for lk in rescaled]
            ks_res = stats.kstest(u_coords, "uniform")
            p_values.append(ks_res.pvalue)
            if ks_res.statistic < d_crit and ks_res.pvalue > 0.01:
                pass_count += 1

        pass_rate = pass_count / 15.0
        assert (
            pass_rate >= 0.86
        ), f"Monte Carlo pass rate {pass_rate:.2%} below 86% threshold (passes={pass_count}/15)"
        median_p = float(np.median(p_values))
        assert (
            0.10 <= median_p <= 0.90
        ), f"Median p-value {median_p:.4f} exhibits severe distributional skew"


# ============================================================================
# 2. Refractory Dead-Time Boundary Stress Tests
# ============================================================================


class TestRefractoryDeadTimeStress:
    """Rigorous boundary and stress tests for absolute refractory dead-time suppression."""

    def test_refractory_exact_boundary_sub_microsecond(self) -> None:
        """Test delta t around exact boundary t_last + refractory_sec."""
        trigger = PoissonSpindleTrigger(
            lambda_0=1e6, idle_min=0.0, tau=1.0, refractory_sec=10.0
        )
        t_last = 500.0

        # Strictly before dead-time: must be False 100% of the time
        epsilons = [1e-9, 1e-6, 1e-3, 0.01, 0.1, 1.0, 5.0, 9.999999]
        for eps in epsilons:
            t_curr = t_last + 10.0 - eps
            res = trigger.should_trigger(
                current_time=t_curr,
                last_active_time=0.0,
                last_dream_time=t_last,
                fatigue=0.0,
                tom_urgency=10.0,
            )
            assert (
                res is False
            ), f"Refractory breach at t_curr={t_curr} (delta={10.0 - eps})"

        # At boundary t_curr = t_last + 10.0: gate released, with lambda_0=1e6 should trigger True
        t_curr_boundary = t_last + 10.0
        res_boundary = trigger.should_trigger(
            current_time=t_curr_boundary,
            last_active_time=0.0,
            last_dream_time=t_last,
            fatigue=0.0,
            tom_urgency=10.0,
        )
        assert res_boundary is True, "Gate failed to release at t_curr = t_last + 10.0"

    def test_refractory_100k_calls_strictly_suppressed(self) -> None:
        """100,000 stochastic calls strictly within refractory period must return False 100%."""
        trigger = PoissonSpindleTrigger(
            lambda_0=1e6, idle_min=0.0, tau=1.0, refractory_sec=10.0
        )
        t_last = 1000.0
        n_calls = 100000
        failures = 0

        for _ in range(n_calls):
            # Sample delta uniformly in [0.0, 9.999999)
            delta = random.uniform(0.0, 9.999999)
            t_curr = t_last + delta
            res = trigger.should_trigger(
                current_time=t_curr,
                last_active_time=0.0,
                last_dream_time=t_last,
                fatigue=0.0,
                tom_urgency=100.0,
            )
            if res is not False:
                failures += 1

        assert (
            failures == 0
        ), f"Refractory violation in {failures}/{n_calls} calls within t < t_last + 10.0"

    def test_refractory_clock_rollback_and_negative_delta(self) -> None:
        """Clock rollback / NTP skew (current_time < last_dream_time) must return False."""
        trigger = PoissonSpindleTrigger(
            lambda_0=1e6, idle_min=0.0, tau=1.0, refractory_sec=10.0
        )
        t_last = 1000.0

        for delta in [-1000.0, -100.0, -1.0, -1e-6, 0.0]:
            t_curr = t_last + delta
            res = trigger.should_trigger(
                current_time=t_curr,
                last_active_time=0.0,
                last_dream_time=t_last,
                fatigue=0.0,
                tom_urgency=5.0,
            )
            assert (
                res is False
            ), f"Clock rollback breached refractory check: delta={delta}"


# ============================================================================
# 3. Extreme Rate Conditions & Numerical Robustness Stress Tests
# ============================================================================


class TestExtremeRatesAndNumericalRobustnessStress:
    """Evaluates edge conditions: rate=0.0, rate=10^6, negative inputs, and NaN/Inf handling."""

    @pytest.fixture
    def trigger(self) -> PoissonSpindleTrigger:
        return PoissonSpindleTrigger(
            lambda_0=0.1, idle_min=15.0, tau=5.0, refractory_sec=10.0
        )

    def test_zero_rate_conditions(self, trigger: PoissonSpindleTrigger) -> None:
        """Verify that zero rate returns infinite interval and should_trigger returns False."""
        # sample_next_interval with rate 0.0
        assert trigger.sample_next_interval(0.0) == float("inf")
        assert trigger.sample_next_interval(-0.0) == float("inf")

        # compute_rate yielding 0.0 via complete fatigue
        rate_fatigued = trigger.compute_rate(
            current_time=100.0, last_active_time=50.0, fatigue=1.0, tom_urgency=1.0
        )
        assert rate_fatigued == 0.0

        # should_trigger with rate = 0.0 must never fire
        res = trigger.should_trigger(
            current_time=100.0,
            last_active_time=50.0,
            last_dream_time=0.0,
            fatigue=1.0,
            tom_urgency=1.0,
        )
        assert res is False

    def test_ultra_high_rate_1e6(self) -> None:
        """Verify rate = 10^6 behaves predictably without overflow or zero division."""
        trigger = PoissonSpindleTrigger(
            lambda_0=1e6, idle_min=0.0, tau=1.0, refractory_sec=10.0
        )

        # sample_next_interval with rate = 1e6
        intervals = [trigger.sample_next_interval(1e6) for _ in range(100)]
        assert all(dt > 0.0 for dt in intervals)
        assert all(math.isfinite(dt) for dt in intervals)
        assert (
            np.mean(intervals) < 1e-4
        ), f"Expected mean < 1e-4s, got {np.mean(intervals)}"

        # should_trigger with massive rate
        assert (
            trigger.should_trigger(
                current_time=100.0,
                last_active_time=0.0,
                last_dream_time=50.0,
                fatigue=0.0,
                tom_urgency=1.0,
            )
            is True
        )

    def test_negative_rate_inputs(self, trigger: PoissonSpindleTrigger) -> None:
        """Verify negative rate inputs to sample_next_interval and compute_rate."""
        # Negative rates to sample_next_interval
        for neg_rate in [-0.001, -1.0, -100.0, -1e6, float("-inf")]:
            assert trigger.sample_next_interval(neg_rate) == float("inf")

        # Negative lambda_0 in trigger
        neg_trigger = PoissonSpindleTrigger(lambda_0=-5.0)
        r = neg_trigger.compute_rate(100.0, 50.0)
        assert r == 0.0

        # Negative urgency
        r_neg_urgency = trigger.compute_rate(100.0, 50.0, tom_urgency=-2.0)
        assert r_neg_urgency == 0.0

    def test_nan_inf_inputs_handling(self, trigger: PoissonSpindleTrigger) -> None:
        """Verify NaN/Inf inputs do not raise unhandled runtime crashes."""
        # sample_next_interval
        assert math.isnan(trigger.sample_next_interval(float("nan")))
        assert trigger.sample_next_interval(float("inf")) == 0.0
        assert trigger.sample_next_interval(float("-inf")) == float("inf")

        # compute_rate with NaN / Inf
        rate_nan_fatigue = trigger.compute_rate(100.0, 50.0, fatigue=float("nan"))
        assert rate_nan_fatigue == 0.0

        rate_nan_urgency = trigger.compute_rate(100.0, 50.0, tom_urgency=float("nan"))
        assert rate_nan_urgency == 0.0

        rate_inf_time = trigger.compute_rate(float("inf"), 50.0)
        assert math.isfinite(rate_inf_time) and rate_inf_time >= 0.0

        # should_trigger with NaN current_time safely evaluates to False
        res_nan_time = trigger.should_trigger(
            current_time=float("nan"),
            last_active_time=50.0,
            last_dream_time=50.0,
        )
        assert res_nan_time is False

    def test_sigmoid_saturation_and_zero_division_guard(self) -> None:
        """Verify tau <= 0 and extreme z values are protected against ZeroDivision / Overflow."""
        # tau = 0.0 protected by max(1e-5, self.tau)
        trigger_zero_tau = PoissonSpindleTrigger(tau=0.0)
        r_zero_tau = trigger_zero_tau.compute_rate(100.0, 50.0)
        assert math.isfinite(r_zero_tau) and r_zero_tau > 0.0

        # Negative tau
        trigger_neg_tau = PoissonSpindleTrigger(tau=-5.0)
        r_neg_tau = trigger_neg_tau.compute_rate(100.0, 50.0)
        assert math.isfinite(r_neg_tau) and r_neg_tau > 0.0

        # Huge z values
        trigger_normal = PoissonSpindleTrigger()
        # Large positive z (idle_duration >> idle_min): z >= 50 -> sigmoid = 1.0
        r_huge_pos = trigger_normal.compute_rate(1e9, 0.0)
        assert pytest.approx(r_huge_pos, rel=1e-5) == 0.1

        # At idle_duration = 0 with default params: z = (0 - 15) / 5 = -3
        # sigmoid(-3) ~= 0.04742587
        r_zero_idle = trigger_normal.compute_rate(100.0, 100.0)
        expected_zero_idle = 0.1 / (1.0 + math.exp(3.0))
        assert pytest.approx(r_zero_idle, rel=1e-5) == expected_zero_idle

        # Deep negative z <= -50 (e.g. idle_min=300, tau=5 -> z = -60):
        # sigmoid = 0.0 -> rate = 0.0
        trigger_deep_gate = PoissonSpindleTrigger(idle_min=300.0, tau=5.0)
        r_deep_neg = trigger_deep_gate.compute_rate(100.0, 100.0)
        assert r_deep_neg == 0.0


# ============================================================================
# 4. Property-Based Stress Fuzzing via Hypothesis
# ============================================================================


class TestPropertyBasedStressHypothesis:
    """Hypothesis-driven property-based fuzzing of rate computation and trigger gating."""

    @settings(max_examples=200, deadline=None)
    @given(
        current_time=st.floats(min_value=0.0, max_value=1e8, allow_nan=False),
        last_active=st.floats(min_value=0.0, max_value=1e8, allow_nan=False),
        fatigue=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        tom_urgency=st.floats(min_value=-10.0, max_value=100.0, allow_nan=False),
    )
    def test_compute_rate_invariants(
        self,
        current_time: float,
        last_active: float,
        fatigue: float,
        tom_urgency: float,
    ) -> None:
        """Compute rate must always return a finite, non-negative float."""
        trigger = PoissonSpindleTrigger()
        rate = trigger.compute_rate(
            current_time, last_active, fatigue=fatigue, tom_urgency=tom_urgency
        )
        assert isinstance(rate, float)
        assert not math.isnan(rate)
        assert rate >= 0.0

    @settings(max_examples=200, deadline=None)
    @given(
        t_curr=st.floats(min_value=100.0, max_value=1000.0, allow_nan=False),
        delta=st.floats(min_value=0.0, max_value=9.9999, allow_nan=False),
    )
    def test_refractory_invariant_property(self, t_curr: float, delta: float) -> None:
        """Within refractory period, should_trigger must be unconditionally False."""
        trigger = PoissonSpindleTrigger(lambda_0=1e5, refractory_sec=10.0)
        t_last_dream = t_curr - delta
        res = trigger.should_trigger(
            current_time=t_curr,
            last_active_time=0.0,
            last_dream_time=t_last_dream,
            fatigue=0.0,
            tom_urgency=5.0,
        )
        assert res is False
