from __future__ import annotations

import argparse
from pathlib import Path

from .algorithms import ALGORITHMS
from .benchmark import run_benchmark, summarize, write_results
from .problem import load_klia_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare gate-allocation metaheuristics")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    problem = load_klia_case()
    results = run_benchmark(
        problem,
        args.algorithms,
        args.runs,
        args.population,
        args.iterations,
        args.seed,
    )
    write_results(results, problem, args.output)
    print("algorithm  runs          best          mean           std         worst")
    for row in summarize(results):
        print(
            f"{row['algorithm']:<9} {row['runs']:>4} "
            f"{row['best']:>13.1f} {row['mean']:>13.1f} "
            f"{row['std']:>13.1f} {row['worst']:>13.1f}"
        )
    print(f"\nResults written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
