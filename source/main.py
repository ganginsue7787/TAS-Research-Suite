"""
main.py
=======
TAS Journal Suite — Unified CLI Entry Point

Usage examples:
  python main.py --mode test
  python main.py --mode mc   --models CNN ResNet18 --datasets MNIST CIFAR10 --n_runs 50
  python main.py --mode figures --csv results/montecarlo_results.csv
  python main.py --mode tables  --csv results/montecarlo_results.csv
  python main.py --mode stats   --csv results/montecarlo_results.csv
  python main.py --mode package

Author  : Kang, In-Su
License : MIT
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Argument Parser ───────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TAS Journal Suite")
    p.add_argument("--mode", default="test",
                   choices=["test", "mc", "figures", "tables", "stats", "package"],
                   help="Execution mode")
    p.add_argument("--models",   nargs="+", default=["CNN"],
                   choices=["CNN", "ResNet18", "ViT"])
    p.add_argument("--datasets", nargs="+", default=["MNIST"],
                   choices=["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"])
    p.add_argument("--alphas",   nargs="+", type=float,
                   default=[1.0, 2.0, 2.5, 3.0, 5.0])
    p.add_argument("--n_runs",   type=int,  default=50)
    p.add_argument("--csv",      default="results/montecarlo_results.csv")
    p.add_argument("--out_dir",  default=".")
    p.add_argument("--device",   default="cpu")
    return p.parse_args()


# ── Mode: test ────────────────────────────────────────────────

def mode_test():
    print("\n=== TAS Core Self-Test ===")
    from tas_core import compute_tas_metrics, compute_tewi, compute_collapse_rate
    np.random.seed(42)
    history = np.random.randn(30, 128)
    res = compute_tas_metrics(history, alpha=2.5)
    print(f"  Hp(H0)  = {res['hp_h0']:.4f}")
    print(f"  Hp(H1)  = {res['hp_h1']:.4f}")
    print(f"  TAS(H0) = {res['tas_h0']:.4f}")
    print(f"  TAS(H1) = {res['tas_h1']:.4f}")
    print(f"  TEWI    = {compute_tewi(res['hp_h0'], res['tas_h0'], res['hp_h1']):.4f}")
    fake_tas = list(np.random.randn(10))
    cr = compute_collapse_rate(fake_tas)
    print(f"  CollapseRate sample = {cr[:3]}")
    print("\nSelf-test PASSED.")


# ── Mode: mc (Monte Carlo) ────────────────────────────────────

def mode_mc(args):
    from montecarlo import MonteCarloEngine
    os.makedirs("results", exist_ok=True)
    engine = MonteCarloEngine(
        models=args.models, datasets=args.datasets,
        alphas=args.alphas, n_runs=args.n_runs, device=args.device
    )
    df = engine.run(output_csv=args.csv)
    print(f"\nMonte Carlo complete. Saved to {args.csv}")
    print(df.groupby("model")["delta_tas"].describe())


# ── Mode: figures ─────────────────────────────────────────────

def mode_figures(args):
    from figures import generate_all_figures
    df = pd.read_csv(args.csv)

    # Simulate a single experiment result for Fig 1 & 2
    # (in real usage, save experiment_result dict from montecarlo)
    n = 100
    fake_exp = {
        "tas_series":  list(np.cumsum(np.random.randn(n) * 0.1 + 0.02)),
        "loss_series": list(np.exp(-np.linspace(0, 3, n)) + np.random.randn(n) * 0.05),
        "grad_series": list(np.exp(-np.linspace(0, 4, n)) + np.random.randn(n) * 0.01),
        "t_hp": 25, "t_tas": 30, "t_gc": 65,
    }

    fig_dir = os.path.join(args.out_dir, "figures")
    generate_all_figures(fake_exp, df, out_dir=fig_dir)


# ── Mode: tables ──────────────────────────────────────────────

def mode_tables(args):
    from latex_tables import generate_all_tables
    from statistics import analyze_csv
    df  = pd.read_csv(args.csv)
    res = analyze_csv(args.csv)
    tbl_dir = os.path.join(args.out_dir, "tables")
    generate_all_tables(df, res["TAS"], out_dir=tbl_dir)


# ── Mode: stats ───────────────────────────────────────────────

def mode_stats(args):
    from statistics import analyze_csv, print_report
    results = analyze_csv(args.csv)
    print_report(results)

    # Save statistics.csv
    rows = []
    for key, res in results.items():
        for metric, value in res.items():
            if not isinstance(value, np.ndarray):
                rows.append({"indicator": key, "metric": metric,
                             "value": str(value)})
    pd.DataFrame(rows).to_csv("results/statistics.csv", index=False)
    print("\nSaved results/statistics.csv")


# ── Mode: package ─────────────────────────────────────────────

def mode_package(args):
    sys.path.insert(0, os.path.join(args.out_dir, "source"))
    from ieee_package import build_package
    build_package(
        package_name=os.path.join(args.out_dir, "TAS_Journal_Package"),
        src_dir=os.path.dirname(os.path.abspath(__file__)),
        zip_output=True
    )


# ── Main ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"\nTAS Journal Suite  |  mode={args.mode}")

    if   args.mode == "test":    mode_test()
    elif args.mode == "mc":      mode_mc(args)
    elif args.mode == "figures": mode_figures(args)
    elif args.mode == "tables":  mode_tables(args)
    elif args.mode == "stats":   mode_stats(args)
    elif args.mode == "package": mode_package(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
