"""
Standalone CLI Test Runner for the Quanta 4-Tier E2E Test Suite.

Executes and verifies:
  - Tier 1: Feature Coverage (80 tests across 16 core features)
  - Tier 2: Boundary & Corner Cases (40 tests across edge conditions)
  - Tier 3: Cross-Feature Combinations (16 tests across pairwise subsystems)
  - Tier 4: Real-World Application Scenarios (6 tests across complex workflows)

Usage:
  python tests/e2e/runner.py
  python tests/e2e/runner.py --tier 1
  python tests/e2e/runner.py --tier 4 -v
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pytest

E2E_DIR = Path(__file__).resolve().parent

TIER_FILES = {
    1: E2E_DIR / "test_tier1_features.py",
    2: E2E_DIR / "test_tier2_boundaries.py",
    3: E2E_DIR / "test_tier3_combinations.py",
    4: E2E_DIR / "test_tier4_applications.py",
}

TIER_NAMES = {
    1: "Tier 1: Feature Coverage (R1, R2, R3, R4)",
    2: "Tier 2: Boundary & Corner Cases",
    3: "Tier 3: Cross-Feature Interactions",
    4: "Tier 4: Real-World Application Scenarios",
}


def run_e2e_suite(tier: int | None = None, verbose: bool = False) -> int:
    """Executes the E2E test suite and returns exit code."""
    print("=" * 80)
    print("QUANTA SDK: 4-TIER OPAQUE-BOX E2E TEST SUITE RUNNER")
    print("=" * 80)

    selected_tiers = [tier] if tier is not None else [1, 2, 3, 4]
    total_exit_code = 0
    tier_results = {}
    start_time_all = time.perf_counter()

    for t in selected_tiers:
        test_file = TIER_FILES[t]
        name = TIER_NAMES[t]
        print(f"\n>>> Running {name}...")
        print(f"    Target: {test_file.name}")

        pytest_args = [
            str(test_file),
            "--no-cov",
            "-q" if not verbose else "-v",
            "--tb=short",
        ]

        t0 = time.perf_counter()
        code = pytest.main(pytest_args)
        elapsed = time.perf_counter() - t0

        status_str = "PASSED" if code == 0 else f"FAILED (code {code})"
        tier_results[t] = (status_str, elapsed)

        if code != 0:
            total_exit_code = code

    total_elapsed = time.perf_counter() - start_time_all

    print("\n" + "=" * 80)
    print("E2E TEST EXECUTION SUMMARY")
    print("=" * 80)
    for t in selected_tiers:
        status, elapsed = tier_results[t]
        print(f"  {TIER_NAMES[t]:<50} : [{status}] in {elapsed:.2f}s")
    print("-" * 80)
    print(f"Total Execution Time: {total_elapsed:.2f}s")
    status_text = "ALL TIERS PASSED (100%)" if total_exit_code == 0 else "TEST FAILURES DETECTED"
    print(f"Overall Status: {status_text}")
    print("=" * 80)

    return total_exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Quanta E2E Test Suite")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run only a specific tier (1, 2, 3, or 4)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output")
    args = parser.parse_args()

    sys.exit(run_e2e_suite(tier=args.tier, verbose=args.verbose))


if __name__ == "__main__":
    main()
