from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .algorithms import ALGORITHMS, RunResult
from .problem import AirportGateProblem


def run_benchmark(
    problem: AirportGateProblem,
    algorithm_names: list[str],
    runs: int,
    population: int,
    iterations: int,
    base_seed: int = 42,
) -> list[RunResult]:
    unknown = sorted(set(algorithm_names) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown algorithms: {', '.join(unknown)}")
    results: list[RunResult] = []
    for run in range(runs):
        seed = base_seed + run
        for name in algorithm_names:
            results.append(ALGORITHMS[name](problem, population, iterations, seed))
    return results


def summarize(results: list[RunResult]) -> list[dict[str, float | int | str]]:
    rows = []
    for name in sorted({result.algorithm for result in results}):
        scores = np.array([result.best_score for result in results if result.algorithm == name])
        rows.append(
            {
                "algorithm": name,
                "runs": int(scores.size),
                "best": float(np.min(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "worst": float(np.max(scores)),
            }
        )
    return sorted(rows, key=lambda row: float(row["mean"]))


def write_results(results: list[RunResult], problem: AirportGateProblem, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["algorithm", "seed", "score", "evaluations"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "algorithm": result.algorithm,
                    "seed": result.seed,
                    "score": result.best_score,
                    "evaluations": result.evaluations,
                }
            )
    summary = summarize(results)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    best = min(results, key=lambda result: result.best_score)
    (output_dir / "best_assignment.json").write_text(
        json.dumps(
            {
                "algorithm": best.algorithm,
                "seed": best.seed,
                "score": best.best_score,
                "assignment": best.best_assignment.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "best_schedule.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["flight", "start", "end", "passengers", "gate"],
        )
        writer.writeheader()
        for flight, gate in zip(problem.flights, best.best_assignment, strict=True):
            writer.writerow(
                {
                    "flight": flight.code,
                    "start": flight.start,
                    "end": flight.end,
                    "passengers": flight.passengers,
                    "gate": int(gate),
                }
            )
