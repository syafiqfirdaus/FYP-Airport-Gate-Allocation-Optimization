# Airport Gate Allocation Optimization

A reproducible Python comparison of six population-based optimizers for the Airport Gate
Allocation Problem (AGAP):

- Gravitational Search Algorithm (GSA)
- Binary Gravitational Search Algorithm (BGSA)
- Particle Swarm Optimization (PSO)
- Binary Particle Swarm Optimization (BPSO)
- Simulated Kalman Filter (SKF)
- Binary Simulated Kalman Filter (BSKF)

The project converts and modernizes the supplied MATLAB GSA/BGSA implementations, implements
the remaining methods from their primary papers, and applies all six to one common 40-flight,
16-gate case reconstructed from the degree FYP materials.

## What is included

- One objective, constraint handler, evaluation budget, and seed schedule for all methods
- Continuous and four-bit-per-flight binary representations
- Hard conflict checking against nine fixed first flights
- CSV summaries and a JSON best schedule
- Unit tests for feasibility and all six optimizers
- Explicit documentation of source-data limitations and research references

The raw thesis, slides, and papers are not committed. They contain a personal identity number
and may include source data supplied by an airline. The normalized case uses only flight codes,
coarse time windows, passenger counts, gate indices, and walking distances needed for the model.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m agap_opt --runs 10 --population 50 --iterations 500
```

For a quick smoke benchmark:

```powershell
python -m agap_opt --runs 2 --population 10 --iterations 20
```

Outputs are written to `results/`:

- `runs.csv`: one row per algorithm and seed
- `summary.csv`: best, mean, standard deviation, and worst score
- `best_assignment.json`: the best gate vector found
- `best_schedule.csv`: the best assignment with flight times and passenger counts

An illustrative five-run comparison is checked into `examples/sample-results/`. It uses only
20 agents and 50 iterations, so it demonstrates the workflow rather than supporting a final
scientific ranking. Run the larger command above (or a preregistered budget) for thesis results.

Lower scores are better. The algorithms are stochastic; use repeated runs and report the full
configuration. A good result is not a proof of the global optimum.

## Model scope

The implemented objective is passenger-weighted walking distance between each assigned gate and
the terminal entrance. Gate conflicts are forbidden. The fixed-flight contribution is included in
reported totals. Transfer-passenger terms mentioned in the thesis are not included because the
complete connection mapping could not be recovered unambiguously from the supplied artifacts.

See [methodology](docs/methodology.md) for representation, repair, comparison rules, and the
documented 40-versus-41-flight source discrepancy. See [references](docs/references.md) for the
papers behind each algorithm.

## Repository layout

```text
src/agap_opt/problem.py     Reconstructed case, repair, objective, validation
src/agap_opt/algorithms.py  GSA, BGSA, PSO, BPSO, SKF, BSKF
src/agap_opt/benchmark.py   Repeated runs and result export
src/agap_opt/cli.py         Command-line interface
tests/                      Feasibility and optimizer smoke tests
docs/                       Methodology and primary references
```

## Responsible reuse

Before publishing extensions, confirm that any additional airline or thesis data may legally be
shared. Do not commit identity-card numbers, student records, or confidential operational datasets.
