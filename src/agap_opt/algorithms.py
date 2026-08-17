from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .problem import AirportGateProblem


@dataclass(frozen=True)
class RunResult:
    algorithm: str
    best_score: float
    best_assignment: np.ndarray
    history: np.ndarray
    evaluations: int
    seed: int


def _evaluate(problem: AirportGateProblem, candidates: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    evaluations = [problem.evaluate_preferences(candidate) for candidate in candidates]
    return np.array([item.score for item in evaluations]), [item.assignment for item in evaluations]


def _mass(fitness: np.ndarray) -> np.ndarray:
    best, worst = float(np.min(fitness)), float(np.max(fitness))
    if np.isclose(best, worst):
        return np.full(fitness.size, 1.0 / fitness.size)
    raw = (fitness - worst) / (best - worst)
    total = float(np.sum(raw))
    return raw / total if total > 0 else np.full(fitness.size, 1.0 / fitness.size)


def _finish(name: str, best_score: float, best_assignment: np.ndarray, history: list[float], population: int, seed: int) -> RunResult:
    return RunResult(name, best_score, best_assignment.copy(), np.asarray(history), population * len(history), seed)


def gsa(problem: AirportGateProblem, population: int, iterations: int, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    n, d = population, problem.dimension
    x = rng.uniform(0, problem.n_gates - 1, (n, d))
    velocity = np.zeros_like(x)
    best_score = np.inf
    best_assignment = np.zeros(d, dtype=int)
    history: list[float] = []
    for step in range(iterations):
        fitness, assignments = _evaluate(problem, np.rint(x).astype(int))
        index = int(np.argmin(fitness))
        if fitness[index] < best_score:
            best_score, best_assignment = float(fitness[index]), assignments[index]
        history.append(best_score)
        mass = _mass(fitness)
        gravity = 100.0 * np.exp(-20.0 * (step + 1) / iterations)
        kbest = max(1, round(n * (2 + (1 - (step + 1) / iterations) * 98) / 100))
        leaders = np.argsort(mass)[::-1][:kbest]
        acceleration = np.zeros_like(x)
        for i in range(n):
            for j in leaders:
                if i == j:
                    continue
                distance = np.linalg.norm(x[i] - x[j])
                acceleration[i] += rng.random(d) * mass[j] * (x[j] - x[i]) / (distance + 1e-12)
        acceleration *= gravity
        velocity = rng.random((n, d)) * velocity + acceleration
        x = np.clip(x + velocity, 0, problem.n_gates - 1)
    return _finish("GSA", best_score, best_assignment, history, population, seed)


def pso(problem: AirportGateProblem, population: int, iterations: int, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    n, d = population, problem.dimension
    x = rng.uniform(0, problem.n_gates - 1, (n, d))
    velocity = np.zeros_like(x)
    personal_x, personal_scores = x.copy(), np.full(n, np.inf)
    global_x, best_score = x[0].copy(), np.inf
    best_assignment = np.zeros(d, dtype=int)
    history: list[float] = []
    for step in range(iterations):
        scores, assignments = _evaluate(problem, np.rint(x).astype(int))
        improved = scores < personal_scores
        personal_x[improved], personal_scores[improved] = x[improved], scores[improved]
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            global_x, best_score, best_assignment = x[index].copy(), float(scores[index]), assignments[index]
        history.append(best_score)
        inertia = 0.9 - 0.5 * step / max(iterations - 1, 1)
        velocity = (
            inertia * velocity
            + 2.0 * rng.random((n, d)) * (personal_x - x)
            + 2.0 * rng.random((n, d)) * (global_x - x)
        )
        x = np.clip(x + velocity, 0, problem.n_gates - 1)
    return _finish("PSO", best_score, best_assignment, history, population, seed)


def skf(problem: AirportGateProblem, population: int, iterations: int, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    n, d = population, problem.dimension
    x = rng.uniform(0, problem.n_gates - 1, (n, d))
    covariance = np.ones((n, d))
    process_noise, measurement_noise = 0.5, 0.5
    best_score = np.inf
    true_state = x[0].copy()
    best_assignment = np.zeros(d, dtype=int)
    history: list[float] = []
    for _ in range(iterations):
        scores, assignments = _evaluate(problem, np.rint(x).astype(int))
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            best_score, true_state, best_assignment = float(scores[index]), x[index].copy(), assignments[index]
        history.append(best_score)
        predicted = x
        predicted_covariance = covariance + process_noise
        measurement = predicted + np.sin(rng.random((n, d)) * 2 * np.pi) * np.abs(predicted - true_state)
        gain = predicted_covariance / (predicted_covariance + measurement_noise)
        x = np.clip(predicted + gain * (measurement - predicted), 0, problem.n_gates - 1)
        covariance = (1 - gain) * predicted_covariance
    return _finish("SKF", best_score, best_assignment, history, population, seed)


def _decode(bits: np.ndarray) -> np.ndarray:
    reshaped = bits.reshape(bits.shape[0], -1, 4)
    return np.sum(reshaped * np.array([8, 4, 2, 1]), axis=2).astype(int)


def bgsa(problem: AirportGateProblem, population: int, iterations: int, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    n, bit_count = population, problem.dimension * 4
    x = rng.integers(0, 2, (n, bit_count), dtype=np.int8)
    velocity = np.zeros((n, bit_count))
    best_score = np.inf
    best_assignment = np.zeros(problem.dimension, dtype=int)
    history: list[float] = []
    previous_x, previous_scores, previous_assignments = None, None, None
    for step in range(iterations):
        scores, assignments = _evaluate(problem, _decode(x))
        if previous_scores is not None:
            worse = scores > previous_scores
            x[worse], scores[worse] = previous_x[worse], previous_scores[worse]
            for index in np.flatnonzero(worse):
                assignments[int(index)] = previous_assignments[int(index)]
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            best_score, best_assignment = float(scores[index]), assignments[index]
        history.append(best_score)
        mass = _mass(scores)
        gravity = 1.0 - (step + 1) / iterations
        kbest = max(1, round(n * (2 + (1 - (step + 1) / iterations) * 98) / 100))
        leaders = np.argsort(mass)[::-1][:kbest]
        acceleration = np.zeros_like(velocity)
        for i in range(n):
            for j in leaders:
                if i == j:
                    continue
                hamming = np.mean(x[i] != x[j])
                acceleration[i] += rng.random(bit_count) * mass[j] * (x[j] - x[i]) / (hamming + 1 / bit_count)
        acceleration *= gravity
        previous_x, previous_scores = x.copy(), scores.copy()
        previous_assignments = [assignment.copy() for assignment in assignments]
        velocity = rng.random((n, bit_count)) * velocity + acceleration
        flips = rng.random((n, bit_count)) < np.abs(np.tanh(velocity))
        x = np.logical_xor(x, flips).astype(np.int8)
    return _finish("BGSA", best_score, best_assignment, history, population, seed)


def bpso(problem: AirportGateProblem, population: int, iterations: int, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    n, bit_count = population, problem.dimension * 4
    x = rng.integers(0, 2, (n, bit_count), dtype=np.int8)
    velocity = np.zeros((n, bit_count))
    personal_x, personal_scores = x.copy(), np.full(n, np.inf)
    global_x, best_score = x[0].copy(), np.inf
    best_assignment = np.zeros(problem.dimension, dtype=int)
    history: list[float] = []
    for step in range(iterations):
        scores, assignments = _evaluate(problem, _decode(x))
        improved = scores < personal_scores
        personal_x[improved], personal_scores[improved] = x[improved], scores[improved]
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            global_x, best_score, best_assignment = x[index].copy(), float(scores[index]), assignments[index]
        history.append(best_score)
        inertia = 0.9 - 0.5 * step / max(iterations - 1, 1)
        velocity = (
            inertia * velocity
            + 2.0 * rng.random((n, bit_count)) * (personal_x - x)
            + 2.0 * rng.random((n, bit_count)) * (global_x - x)
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(velocity, -30, 30)))
        x = (rng.random((n, bit_count)) < probability).astype(np.int8)
    return _finish("BPSO", best_score, best_assignment, history, population, seed)


def bskf(problem: AirportGateProblem, population: int, iterations: int, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    n, bit_count = population, problem.dimension * 4
    x = rng.integers(0, 2, (n, bit_count), dtype=np.int8)
    covariance = np.ones((n, bit_count))
    process_noise, measurement_noise = 0.5, 0.5
    best_score = np.inf
    true_state = x[0].copy()
    best_assignment = np.zeros(problem.dimension, dtype=int)
    history: list[float] = []
    for _ in range(iterations):
        scores, assignments = _evaluate(problem, _decode(x))
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            best_score, true_state, best_assignment = float(scores[index]), x[index].copy(), assignments[index]
        history.append(best_score)
        predicted = x.astype(float)
        predicted_covariance = covariance + process_noise
        measurement = predicted + np.sin(rng.random((n, bit_count)) * 2 * np.pi) * np.abs(predicted - true_state)
        gain = predicted_covariance / (predicted_covariance + measurement_noise)
        innovation = gain * (measurement - predicted)
        flips = rng.random((n, bit_count)) < np.abs(np.tanh(innovation))
        x = np.logical_xor(x, flips).astype(np.int8)
        covariance = (1 - gain) * predicted_covariance
    return _finish("BSKF", best_score, best_assignment, history, population, seed)


ALGORITHMS: dict[str, Callable[[AirportGateProblem, int, int, int], RunResult]] = {
    "gsa": gsa, "bgsa": bgsa, "pso": pso, "bpso": bpso, "skf": skf, "bskf": bskf
}
