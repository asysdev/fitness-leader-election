"""
run_all_experiments.py — Master script to run all 5 experiments sequentially.

Usage:
    python scripts/run_all_experiments.py [--quick]

Options:
    --quick     Use reduced trial counts for fast testing (~2 min).
                Default is full run (~20-40 min depending on hardware).
"""

import sys
import os
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Argument parsing ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run all fitness leader election experiments")
parser.add_argument("--quick", action="store_true",
                    help="Reduced run for smoke-testing (~2 min)")
args = parser.parse_args()

if args.quick:
    # Patch experiment configs for fast runs
    os.environ["FLE_QUICK"] = "1"
    print("⚡ Quick mode: using reduced trial counts.\n")

# ── Import experiment modules ─────────────────────────────────────────
import scripts.run_election_quality as exp1
import scripts.run_convergence as exp2
import scripts.run_failure_recovery as exp3
import scripts.run_churn as exp4
import scripts.run_weight_sensitivity as exp5
import scripts.generate_plots as plots
import scripts.generate_tables as tables


def patch_quick_mode():
    """Override N_TRIALS and other counts for quick testing."""
    exp1.N_TRIALS = 5
    exp1.SWARM_SIZES = [10, 30]
    exp2.N_TRIALS = 5
    exp2.SWARM_SIZES = [5, 10, 20, 50]
    exp3.N_TRIALS = 5
    exp3.SWARM_SIZES = [10, 30]
    exp4.N_TRIALS = 3
    exp4.N_ROUNDS = 20
    exp5.N_SAMPLES = 3
    exp5.N_AGENTS = 10


def run_experiment(name: str, fn, idx: int, total: int) -> float:
    """Run one experiment function with timing and error handling."""
    print(f"\n{'='*60}")
    print(f"Experiment {idx}/{total}: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"✓ Completed in {elapsed:.1f}s")
        return elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"✗ FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return elapsed


def main():
    if args.quick:
        patch_quick_mode()

    total_t0 = time.time()

    experiments = [
        ("Election Quality (LQS)", exp1.main),
        ("Convergence Speed", exp2.main),
        ("Failure Recovery", exp3.main),
        ("Node Churn Stability", exp4.main),
        ("Weight Sensitivity", exp5.main),
    ]

    timings = {}
    for i, (name, fn) in enumerate(experiments, 1):
        elapsed = run_experiment(name, fn, i, len(experiments))
        timings[name] = elapsed

    # Generate figures and tables
    print(f"\n{'='*60}")
    print("Generating Figures and LaTeX Tables")
    print(f"{'='*60}")
    try:
        plots.main()
        tables.main()
    except Exception as e:
        print(f"✗ Plot/table generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    for name, t in timings.items():
        print(f"  {name:<35} {t:>6.1f}s")
    print(f"  {'─'*45}")
    print(f"  {'Total':.<35} {total_elapsed:>6.1f}s")

    results_dir = Path(__file__).parent.parent / "results"
    print(f"\n✓ Results saved to: {results_dir}/")
    print(f"  CSVs:    {results_dir}/*.csv")
    print(f"  Figures: {results_dir}/figures/")
    print(f"  Tables:  {results_dir}/tables/")


if __name__ == "__main__":
    main()
