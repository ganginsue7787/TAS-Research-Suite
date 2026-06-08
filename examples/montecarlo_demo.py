"""
montecarlo_demo.py
==================
TAS Research Suite — Mini Monte Carlo Demo (3 runs)

Quick reproducibility check for the Monte Carlo engine.
Full experiment: python source/main.py --mode mc --n_runs 50

Run:
    cd TAS_GitHub
    python examples/montecarlo_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

from montecarlo import MonteCarloEngine
from statistics  import analyze_delta_t, print_report
import pandas as pd


def main():
    print('=' * 55)
    print('  TAS Mini Monte Carlo Demo (3 runs × CNN × MNIST)')
    print('=' * 55)
    print('\nNote: For full experiment use --n_runs 50\n')

    engine = MonteCarloEngine(
        models   = ['CNN'],
        datasets = ['MNIST'],
        alphas   = [2.5],
        n_runs   = 3,
        device   = 'cpu',
    )

    df = engine.run(output_csv='examples/montecarlo_demo_results.csv')

    print('\n=== Raw Results ===')
    print(df[['run', 'model', 'dataset', 'alpha',
              'delta_tas', 'delta_hp', 't_gc', 't_tas', 't_hp']].to_string())

    valid = df['delta_tas'].dropna()
    if len(valid) > 1:
        print('\n=== Statistical Summary (delta_tas) ===')
        res = analyze_delta_t(valid)
        print(f'  N          = {res["n"]}')
        print(f'  Mean Δt    = {res["mean"]:.2f}')
        print(f'  Std        = {res["std"]:.2f}')
    else:
        print('\n(Need ≥ 2 valid runs for statistics — increase n_runs)')

    print('\nDemo complete. See examples/montecarlo_demo_results.csv  ✓')


if __name__ == '__main__':
    main()
