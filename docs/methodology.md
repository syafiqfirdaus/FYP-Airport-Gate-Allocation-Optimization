# Methodology

## Problem representation

The reconstructed case has 16 gates, 31 flights whose gates are optimized, and nine
already-present flights with fixed gates. A half-open interval `[arrival, departure)` is
used, so a gate can serve a new flight at the exact departure time of its previous flight.

Each candidate holds one preferred gate per variable flight. Continuous algorithms use
31 real coordinates rounded to gate indices. Binary algorithms use the thesis encoding:
four bits per flight, or 124 bits in total, decoded as an integer from 0 to 15.

## Constraint handling

A shared deterministic repair operator schedules flights chronologically. It uses the
preferred gate when possible and otherwise selects the numerically nearest feasible gate.
This guarantees that all six optimizers are compared using the same feasibility mechanism.
Every returned schedule is checked against both variable and fixed flights.

## Objective

The current reproducible objective minimizes total passenger walking distance to/from the
terminal entrance:

`sum(passengers[f] * entrance_distance[assigned_gate[f]])`

The constant contribution of the nine fixed flights is included in reported totals. The
thesis also describes transfer-passenger terms, but its displayed equation and complete
connection mapping were embedded as figures and are not recoverable unambiguously from the
available source. Those terms are therefore not invented here.

## Comparison protocol

All algorithms receive the same population size, iteration count, run seeds, decoder,
repair operator, and objective. Report multiple independent runs and compare at least the
best, mean, standard deviation, and worst final objective. These stochastic heuristics do
not prove global optimality; conclusions should be restricted to the stated case and budget.

## Source-data decision

The thesis alternates between 40 and 41 total flights. Its main tables and slides specify
31 optimized flights plus nine first flights, while an appendix fragment contains a tenth
fixed record (`TW41`) not present in the main case. This repository follows the internally
consistent 40-flight definition and records the discrepancy for reproducibility.
