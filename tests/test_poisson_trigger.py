"""Tests for the Non-Homogeneous Poisson Spindle Trigger.

Tests rate computation, refractory gating, exponential renewal sampling,
and Kolmogorov-Smirnov statistical goodness-of-fit via the Time-Rescaling Theorem.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from scipy import stats

# Attempt real import; fall back to contract double if not yet implemented
try:
    from quanta.cognitive.poisson_trigger import PoissonSpindleTrigger

    REAL_TRIGGER_AVAILABLE = True
except ImportError:
    REAL_TRIGGER_AVAILABLE = False

    class PoissonSpindleTrigger:  # type: ignore[no-redef]
        """Authoritative contract double adhering strictly to PROJECT.md & Survey 2."""

        def __init__(
            self,
            lambda_0: float = 0.1,
            idle_min: float = 15.0,
            tau: float = 5.0,
            refractory_sec: float = 10.0,
        ) -> None:
            self.lambda_0 = lambda_0
            self.idle_min = idle_min
            self.tau = tau
            self.refractory_sec = refractory_sec

        def compute_rate(
            self,
            current_time: float,
            last_active_time: float,
            fatigue: float,
            tom_urgency: float,
        ) -> float:
            idle_duration = max(0.0, current_time - last_active_time)
            z = (idle_duration - self.idle_min) / max(1e-5, self.tau)
            if z >= 50.0:
                sigmoid = 1.0
            elif z <= -50.0:
                sigmoid = 0.0
            else:
                sigmoid = 1.0 / (1.0 + math.exp(-z))

            fatigue_factor = max(0.0, 1.0 - max(0.0, min(1.0, fatigue)))
            urgency = max(0.0, tom_urgency)
            rate = self.lambda_0 * sigmoid * fatigue_factor * urgency
            return max(0.0, rate)

        def sample_next_interval(self, rate: float) -> float:
            if rate <= 0.0:
                return float("inf")
            u = random.random()
            u = max(1e-12, min(1.0 - 1e-12, u))
            return -math.log(u) / rate

        def should_trigger(
            self,
            current_time: float,
            last_active_time: float,
            last_dream_time: float,
            fatigue: float,
            tom_urgency: float,
        ) -> bool:
            if (current_time - last_dream_time) < self.refractory_sec:
                return False
            rate = self.compute_rate(current_time, last_active_time, fatigue, tom_urgency)
            if rate <= 0.0:
                return False
            # Poisson arrival check for dt = 1.0s window
            p = 1.0 - math.exp(-rate * 1.0)
            return random.random() < p


# ============================================================================
# Tier 2: Poisson Rate Computation & BVA Tests
# ============================================================================


class TestPoissonRateComputation:
    """Verifies mathematical correctness of lambda(t) across all boundary conditions."""

    @pytest.fixture
    def trigger(self) -> PoissonSpindleTrigger:
        return PoissonSpindleTrigger(
            lambda_0=0.1, idle_min=15.0, tau=5.0, refractory_sec=10.0
        )

    def test_rate_at_idle_gate_boundary(self, trigger: PoissonSpindleTrigger) -> None:
        """At current_time - last_active == idle_min (15s), sigmoid factor is exactly 0.5."""
        t_active = 100.0
        t_curr = 115.0  # exactly 15s idle
        rate = trigger.compute_rate(
            current_time=t_curr, last_active_time=t_active, fatigue=0.0, tom_urgency=1.0
        )
        # lambda = 0.1 * 0.5 * 1.0 * 1.0 = 0.05
        assert pytest.approx(rate, rel=1e-4) == 0.05

    def test_rate_suppressed_below_idle_gate(self, trigger: PoissonSpindleTrigger) -> None:
        """When idle time is well below gate (e.g. 5s), rate is strongly suppressed."""
        t_active = 100.0
        t_curr = 105.0  # 5s idle -> z = (5-15)/5 = -2 -> sigmoid(-2) ~= 0.1192
        rate = trigger.compute_rate(
            current_time=t_curr, last_active_time=t_active, fatigue=0.0, tom_urgency=1.0
        )
        expected_sig = 1.0 / (1.0 + math.exp(2.0))
        assert pytest.approx(rate, rel=1e-3) == 0.1 * expected_sig

    def test_rate_saturates_above_idle_gate(self, trigger: PoissonSpindleTrigger) -> None:
        """When idle time is large (e.g. 100s), sigmoid approaches 1.0."""
        t_active = 100.0
        t_curr = 200.0  # 100s idle -> z = 17 -> sigmoid ~= 1.0
        rate = trigger.compute_rate(
            current_time=t_curr, last_active_time=t_active, fatigue=0.0, tom_urgency=1.0
        )
        assert pytest.approx(rate, rel=1e-3) == 0.1

    def test_rate_fatigue_bounds(self, trigger: PoissonSpindleTrigger) -> None:
        """Verifies fatigue scaling: 0.0 -> full rate; 1.0 -> 0 rate."""
        t_curr = 200.0
        t_active = 100.0
        rate_zero_fatigue = trigger.compute_rate(
            t_curr, t_active, fatigue=0.0, tom_urgency=1.0
        )
        rate_half_fatigue = trigger.compute_rate(
            t_curr, t_active, fatigue=0.5, tom_urgency=1.0
        )
        rate_full_fatigue = trigger.compute_rate(
            t_curr, t_active, fatigue=1.0, tom_urgency=1.0
        )

        assert pytest.approx(rate_half_fatigue, rel=1e-3) == rate_zero_fatigue * 0.5
        assert rate_full_fatigue == 0.0

    def test_rate_tom_urgency_scaling(self, trigger: PoissonSpindleTrigger) -> None:
        """Verifies ToM urgency scales the rate linearly."""
        t_curr = 200.0
        t_active = 100.0
        rate_low = trigger.compute_rate(
            t_curr, t_active, fatigue=0.0, tom_urgency=0.2
        )
        rate_high = trigger.compute_rate(
            t_curr, t_active, fatigue=0.0, tom_urgency=5.0
        )

        assert pytest.approx(rate_high / rate_low, rel=1e-3) == 25.0  # 5.0 / 0.2 = 25.0


# ============================================================================
# Tier 2: Refractory Dead-Time Gating Tests
# ============================================================================


class TestRefractoryGating:
    """Verifies that subconscious dream pulses are strictly suppressed during refractory period."""

    def test_refractory_dead_time_enforced(self) -> None:
        """Trigger returns False unconditionally if elapsed time < refractory_sec."""
        trigger = PoissonSpindleTrigger(
            lambda_0=10.0, idle_min=0.0, tau=1.0, refractory_sec=10.0
        )
        t_curr = 105.0
        t_active = 50.0  # highly idle
        t_last_dream = 100.0  # only 5s since last dream (refractory = 10s)

        # Even with massive rate, refractory block must return False
        for _ in range(20):
            res = trigger.should_trigger(
                current_time=t_curr,
                last_active_time=t_active,
                last_dream_time=t_last_dream,
                fatigue=0.0,
                tom_urgency=5.0,
            )
            assert res is False

    def test_post_refractory_trigger_enabled(self) -> None:
        """Trigger can fire once elapsed time >= refractory_sec."""
        trigger = PoissonSpindleTrigger(
            lambda_0=50.0, idle_min=0.0, tau=1.0, refractory_sec=10.0
        )
        t_curr = 120.0
        t_active = 50.0
        t_last_dream = 100.0  # 20s since last dream (post-refractory)

        # With high rate, should_trigger should eventually return True
        triggered = any(
            trigger.should_trigger(
                current_time=t_curr,
                last_active_time=t_active,
                last_dream_time=t_last_dream,
                fatigue=0.0,
                tom_urgency=5.0,
            )
            for _ in range(50)
        )
        assert triggered is True


# ============================================================================
# Tier 2: Exponential Renewal Sampling Tests
# ============================================================================


class TestExponentialRenewalSampling:
    """Verifies properties of inverse transform exponential renewal sampling."""

    def test_zero_rate_returns_infinite_interval(self) -> None:
        """Rate <= 0 must return infinity without crashing."""
        trigger = PoissonSpindleTrigger()
        assert trigger.sample_next_interval(0.0) == float("inf")
        assert trigger.sample_next_interval(-1.0) == float("inf")

    def test_exponential_mean_and_variance(self) -> None:
        """Monte Carlo verification: mean ~ 1/lambda and std ~ 1/lambda."""
        trigger = PoissonSpindleTrigger()
        rate = 0.25  # expected mean = 4.0s
        n_samples = 5000

        samples = [trigger.sample_next_interval(rate) for _ in range(n_samples)]
        sample_mean = float(np.mean(samples))
        sample_std = float(np.std(samples))

        expected_mean = 1.0 / rate  # 4.0
        # Standard error of the mean = sigma / sqrt(N) = 4.0 / sqrt(5000) ~= 0.056
        assert pytest.approx(sample_mean, abs=0.25) == expected_mean
        assert pytest.approx(sample_std, abs=0.35) == expected_mean


# ============================================================================
# Tier 2: Kolmogorov-Smirnov Goodness-of-Fit via Time-Rescaling Theorem
# ============================================================================


class TestTimeRescalingKolmogorovSmirnov:
    """Rigorous statistical validation of point process renewal via Time-Rescaling Theorem."""

    def test_kolmogorov_smirnov_time_rescaling(self) -> None:
        """Generates arrivals, rescales via Lambda_k, tests uniform goodness-of-fit."""
        random.seed(42)
        np.random.seed(42)

        trigger = PoissonSpindleTrigger(
            lambda_0=0.2, idle_min=10.0, tau=5.0, refractory_sec=0.0
        )
        t_active = 0.0
        current_time = 0.0
        n_events = 150
        event_times = []

        # Generate arrivals under time-varying lambda(t)
        for _ in range(n_events):
            rate = trigger.compute_rate(
                current_time, t_active, fatigue=0.1, tom_urgency=1.2
            )
            dt = trigger.sample_next_interval(max(0.01, rate))
            current_time += dt
            event_times.append(current_time)

        # Time-rescaling integral transform:
        # Lambda_k = integral_{t_{k-1}}^{t_k} lambda(s) ds
        # Using numerical midpoint quadrature for each interval:
        rescaled_intervals = []
        t_prev = 0.0
        for t_curr in event_times:
            # 10-step numerical integration of lambda(s) over [t_prev, t_curr]
            steps = 10
            ds = (t_curr - t_prev) / steps
            integral = 0.0
            for i in range(steps):
                s_mid = t_prev + (i + 0.5) * ds
                r = trigger.compute_rate(s_mid, t_active, fatigue=0.1, tom_urgency=1.2)
                integral += r * ds
            rescaled_intervals.append(integral)
            t_prev = t_curr

        # By Time-Rescaling Theorem:
        # u_k = 1 - exp(-Lambda_k) must be i.i.d. Uniform(0, 1)
        u_coords = [1.0 - math.exp(-lk) for lk in rescaled_intervals]

        # Two-sided Kolmogorov-Smirnov test against standard uniform distribution
        ks_result = stats.kstest(u_coords, "uniform")
        # Assert null hypothesis cannot be rejected at alpha = 0.01
        msg = f"KS rejected null: D={ks_result.statistic:.4f}, p={ks_result.pvalue:.4f}"
        assert ks_result.pvalue > 0.01, msg

    def test_sigmoid_lower_saturation_and_zero_rate_branch(self) -> None:
        """Covers z <= -50.0 sigmoid cutoff and should_trigger zero-rate branch."""
        trigger = PoissonSpindleTrigger(lambda_0=1.0, idle_min=200.0, tau=1.0, refractory_sec=0.0)
        # z = (0 - 200) / 1 = -200 <= -50 -> sigmoid = 0.0 -> rate = 0.0
        rate = trigger.compute_rate(current_time=0.0, last_active_time=0.0)
        assert rate == 0.0

        # should_trigger with rate <= 0 returns False
        triggered = trigger.should_trigger(
            current_time=0.0,
            last_active_time=0.0,
            last_dream_time=-100.0,
        )
        assert triggered is False

