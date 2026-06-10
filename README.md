# Power Generation Scheduling Project

This project solves a power generation scheduling problem using:

1. Gurobi Mixed-Integer Linear Programming (MILP)
2. Genetic Algorithm (nature-inspired metaheuristic)

The model schedules 10 power plant groups over a multi-day period and minimizes fuel, health, operating, startup, and shutdown costs.


For the 24-hour scheduling case, Gurobi obtained an objective value of 4,495,663.46. The Genetic Algorithm obtained a feasible solution with objective value 4,651,780.30. Since the GA penalty was zero, all modeled constraints were satisfied. The optimality gap between the GA and Gurobi solution was 3.47%, showing that the nature-inspired algorithm produced a near-optimal solution.

## Folder structure

```text
power_generation_project/
├── data/
├── src/
├── results/
├── main.py
├── requirements.txt
└── README.md
```

## Setup

Open the folder in Visual Studio or VS Code, then run:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Change the experiment

In `main.py`, edit:

```python
YEAR = 2011
MONTH = 7
START_DAY = 1
END_DAY = 7
```

For a smaller first test, use:

```python
START_DAY = 1
END_DAY = 1
```

## Outputs

Results are saved in the `results/` folder:

- `gurobi_generation.csv`
- `ga_generation.csv`
- `ga_history.csv`
- `comparison.csv`
- generation and convergence plots

## Important modeling notes

- Nuclear plants are forced to stay on.
- Nuclear plants have 80% minimum generation.
- Non-nuclear plants have 1% minimum generation when on.
- Ramp constraints are included in the Gurobi MILP.
- The Genetic Algorithm uses a penalty function for infeasibilities.
- Health-cost data is available for July 2007, so the project reuses that day/hour pattern for July scheduling experiments.
