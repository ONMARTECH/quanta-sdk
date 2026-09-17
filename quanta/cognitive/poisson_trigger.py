"""Non-Homogeneous Poisson Spindle Trigger.

Governs the stochastic initiation of subconscious mind-wandering incubation cycles
mirroring thalamocortical sleep spindles (11-16 Hz) and TRN low-threshold T-type
calcium channel burst kinetics.
"""

from __future__ import annotations

import math
import random


class PoissonSpindleTrigger:
    """Stochastic Non-Homogeneous Poisson renewal spindle trigger.

    Adheres to the Time-Rescaling Theorem (Brown et al., 2002) and clinical
    thalamocortical pacemaking dynamics. Enforces absolute refractory dead-time
    gating and sigmoidal quiescence gating.
    """

    def __init__(
        self,
        lambda_0: float = 0.1,
        idle_min: float = 15.0,
        tau: float = 5.0,
        refractory_sec: float = 10.0,
        dt_tick: float = 1.0,
    ) -> None:
        """Initialize the Poisson spindle trigger parameters.

        Args:
            lambda_0: Baseline asymptotic spindle hazard rate (Hz).
            idle_min: Quiescence gate threshold (seconds) before dreaming disinhibits.
            tau: Sigmoidal relaxation time constant (seconds).
            refractory_sec: Absolute refractory dead-time period (seconds).
            dt_tick: Discrete epoch simulation tick duration (seconds).
        """
        self.lambda_0 = float(lambda_0)
        self.idle_min = float(idle_min)
        self.tau = float(tau)
        self.refractory_sec = float(refractory_sec)
        self.dt_tick = float(dt_tick)

    def compute_rate(
        self,
        current_time: float,
        last_active_time: float,
        fatigue: float = 0.0,
        tom_urgency: float = 1.0,
    ) -> float:
        """Compute the instantaneous Poisson hazard rate lambda(t).

        lambda(t) = lambda_0 * sigma((t_idle - T_idle_min) / tau) * (1 - F_fatigue) * S_ToM

        Args:
            current_time: Current epoch timestamp (seconds).
            last_active_time: Timestamp of last conscious foreground user activity (seconds).
            fatigue: Computational metabolic fatigue factor in [0.0, 1.0].
            tom_urgency: Theory of Mind sociological urgency multiplier in [0.2, 5.0].

        Returns:
            Instantaneous hazard rate lambda(t) >= 0.0 (Hz).
        """
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
        """Draw next inter-arrival interval Delta t via exact inverse transform sampling.

        Delta t = -ln(U) / rate, with U ~ Uniform(0, 1).

        Args:
            rate: Instantaneous Poisson arrival rate lambda (Hz).

        Returns:
            Sampled inter-arrival time in seconds, or float('inf') if rate <= 0.
        """
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
        fatigue: float = 0.0,
        tom_urgency: float = 1.0,
        dt_tick: float | None = None,
    ) -> bool:
        """Evaluate whether a subconscious incubation cycle should initiate.

        Enforces absolute refractory dead-time: if (current_time - last_dream_time)
        < refractory_sec, returns False unconditionally. Otherwise evaluates stochastic
        arrival probability: p = 1 - exp(-lambda(t) * dt_tick) and draws uniform random variate.

        Args:
            current_time: Current epoch timestamp (seconds).
            last_active_time: Timestamp of last foreground activity (seconds).
            last_dream_time: Timestamp when last dream cycle terminated (seconds).
            fatigue: Computational metabolic fatigue factor in [0.0, 1.0].
            tom_urgency: Theory of Mind sociological urgency multiplier in [0.2, 5.0].
            dt_tick: Optional simulation epoch tick window override (seconds).

        Returns:
            True if a subconscious dream pulse is triggered, False otherwise.
        """
        if (current_time - last_dream_time) < self.refractory_sec:
            return False

        rate = self.compute_rate(
            current_time,
            last_active_time,
            fatigue=fatigue,
            tom_urgency=tom_urgency,
        )
        if rate <= 0.0:
            return False

        tick = self.dt_tick if dt_tick is None else dt_tick
        p = 1.0 - math.exp(-rate * tick)
        return random.random() < p
