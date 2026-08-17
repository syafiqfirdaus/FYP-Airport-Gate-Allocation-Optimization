from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Flight:
    code: str
    start: float
    end: float
    passengers: int
    fixed_gate: int | None = None


@dataclass(frozen=True)
class Evaluation:
    score: float
    assignment: np.ndarray


class AirportGateProblem:
    """Passenger walking-distance gate assignment with hard non-overlap constraints.

    Candidate values are gate preferences. A deterministic repair operator assigns each
    flight to the closest currently feasible gate preference. All optimizers therefore
    receive feasible solutions under exactly the same decoder and objective.
    """

    def __init__(
        self,
        gate_distances: np.ndarray,
        flights: list[Flight],
        fixed_flights: list[Flight],
    ) -> None:
        self.gate_distances = np.asarray(gate_distances, dtype=float)
        self.flights = tuple(flights)
        self.fixed_flights = tuple(fixed_flights)
        if not len(self.gate_distances):
            raise ValueError("At least one gate is required")
        if any(f.fixed_gate is None for f in self.fixed_flights):
            raise ValueError("Every fixed flight requires fixed_gate")
        self._validate_fixed_schedule()

    @property
    def n_gates(self) -> int:
        return int(self.gate_distances.size)

    @property
    def dimension(self) -> int:
        return len(self.flights)

    @staticmethod
    def overlaps(a: Flight, b: Flight) -> bool:
        return a.start < b.end and b.start < a.end

    def _validate_fixed_schedule(self) -> None:
        for i, left in enumerate(self.fixed_flights):
            if not 0 <= int(left.fixed_gate) < self.n_gates:
                raise ValueError(f"Invalid fixed gate for {left.code}")
            for right in self.fixed_flights[i + 1 :]:
                if left.fixed_gate == right.fixed_gate and self.overlaps(left, right):
                    raise ValueError(f"Fixed flights {left.code} and {right.code} conflict")

    def repair(self, preferences: np.ndarray) -> np.ndarray:
        preferences = np.asarray(preferences, dtype=int).reshape(-1)
        if preferences.size != self.dimension:
            raise ValueError(f"Expected {self.dimension} preferences")
        preferences = np.clip(preferences, 0, self.n_gates - 1)
        assignment = np.full(self.dimension, -1, dtype=int)
        scheduled: list[tuple[Flight, int]] = [
            (flight, int(flight.fixed_gate)) for flight in self.fixed_flights
        ]
        order = sorted(range(self.dimension), key=lambda i: (self.flights[i].start, -self.flights[i].end))
        for index in order:
            flight = self.flights[index]
            preferred = int(preferences[index])
            feasible = []
            for gate in range(self.n_gates):
                blocked = any(
                    gate == used_gate and self.overlaps(flight, used_flight)
                    for used_flight, used_gate in scheduled
                )
                if not blocked:
                    feasible.append(gate)
            if not feasible:
                raise ValueError(f"No feasible gate for {flight.code}; check case capacity")
            gate = min(feasible, key=lambda g: (abs(g - preferred), g))
            assignment[index] = gate
            scheduled.append((flight, gate))
        return assignment

    def evaluate_preferences(self, preferences: np.ndarray) -> Evaluation:
        assignment = self.repair(preferences)
        variable = sum(
            flight.passengers * self.gate_distances[gate]
            for flight, gate in zip(self.flights, assignment, strict=True)
        )
        fixed = sum(
            flight.passengers * self.gate_distances[int(flight.fixed_gate)]
            for flight in self.fixed_flights
        )
        return Evaluation(float(variable + fixed), assignment)

    def validate_assignment(self, assignment: np.ndarray) -> bool:
        assignment = np.asarray(assignment, dtype=int).reshape(-1)
        if assignment.size != self.dimension:
            return False
        scheduled = [(flight, int(flight.fixed_gate)) for flight in self.fixed_flights]
        scheduled.extend(zip(self.flights, assignment, strict=True))
        return all(
            not (left_gate == right_gate and self.overlaps(left, right))
            for i, (left, left_gate) in enumerate(scheduled)
            for right, right_gate in scheduled[i + 1 :]
        )


def load_klia_case() -> AirportGateProblem:
    """Load the normalized case reconstructed from the FYP thesis appendix."""

    distances = np.array(
        [1091.5, 1328.8, 1650.0, 2015.6, 2405.2, 2809.0, 3221.6, 3640.1] * 2
    )
    rows = [
        ("F0300", 7, 11, 160), ("F0200", 9, 13, 160),
        ("F0100", 12, 15, 160), ("F0400", 14, 16, 100),
        ("F0301", 15, 18, 70), ("F0401", 18, 20, 60),
        ("F0101", 19, 22, 70), ("F0201", 20, 23, 50),
        ("F0402", 21, 25, 100), ("F0102", 24, 25, 100),
        ("F0800", 9, 13, 120), ("F0700", 12, 16, 120),
        ("F0600", 13, 17, 100), ("F0500", 14, 18, 80),
        ("F0900", 18, 19, 20), ("F0801", 20, 22, 50),
        ("F0701", 21, 25, 150), ("F0601", 21, 25, 100),
        ("F0501", 22, 25, 100), ("F0901", 23, 25, 100),
        ("F1000", 8, 16, 130), ("F1100", 10, 17, 130),
        ("F1200", 12, 18, 130), ("F1300", 13, 19, 50),
        ("F1001", 16, 20, 50), ("F1101", 17, 20, 30),
        ("F1201", 19, 23, 50), ("F1301", 20, 25, 100),
        ("F1400", 20, 25, 100), ("F1102", 20, 25, 100),
        ("F1202", 24, 25, 50),
    ]
    fixed_rows = [
        ("F0403", 6, 8, 100, 3), ("F0103", 6, 9, 100, 4),
        ("F0702", 6, 7, 200, 10), ("F0602", 6, 8, 200, 7),
        ("F0502", 6, 10, 150, 8), ("F0902", 6, 11, 150, 12),
        ("F1302", 6, 8, 100, 1), ("F1103", 6, 9, 100, 2),
        ("F1203", 6, 10, 100, 15),
    ]
    flights = [Flight(*row) for row in rows]
    fixed = [Flight(code, start, end, passengers, gate) for code, start, end, passengers, gate in fixed_rows]
    return AirportGateProblem(distances, flights, fixed)
