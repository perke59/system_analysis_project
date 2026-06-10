# Power Generation Scheduling Using Gurobi and Genetic Algorithm

This project solves a power generation scheduling problem using two approaches:

1. Mixed-Integer Linear Programming with Gurobi
2. Genetic Algorithm as a nature-inspired metaheuristic

The problem is based on scheduling 10 power plant groups to satisfy electricity demand while minimizing total generation cost. The objective includes fuel cost, health cost, operating cost, startup cost, and shutdown cost.

## Project Description

The optimization problem is a unit commitment / economic dispatch problem. For each plant and each time period, the model decides:

* whether the plant is on or off,
* how much electricity the plant generates,
* whether the plant starts up,
* whether the plant shuts down.

The Gurobi model is used as the exact MILP-based optimization approach. The Genetic Algorithm is used as an approximate nature-inspired method and is compared with the Gurobi result.

## Main Result

For the 24-hour scheduling case, Gurobi obtained an objective value of 4,495,663.46.

The Genetic Algorithm obtained a feasible solution with objective value 4,651,780.30. Since the GA penalty was zero, all modeled constraints were satisfied. The optimality gap between the GA and Gurobi solution was 3.47%, showing that the nature-inspired algorithm produced a near-optimal solution.

## Folder Structure

```text
power_generation_project/
├── data/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── genetic_algorithm.py
│   ├── gurobi_model.py
│   └── plotting.py
├── results/
├── main.py
├── requirements.txt
└── README.md
```

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Experiment Settings

The default experiment in `main.py` is the 24-hour case:

```python
YEAR = 2011
MONTH = 7
START_DAY = 1
END_DAY = 1
```

This setting is recommended for the main comparison because it gives a clean feasible result for both Gurobi and the Genetic Algorithm.

A larger 7-day experiment can also be tested by changing:

```python
START_DAY = 1
END_DAY = 7
```

For the 7-day case, the project uses daily rolling-horizon Gurobi optimization because the restricted Gurobi license may not allow solving the full 7-day MILP as one large model.

## Gurobi Mode

In `main.py`, the Gurobi mode can be changed:

```python
GUROBI_MODE = "daily"
```

Options:

* `"daily"`: solves each day separately and combines the results
* `"combined"`: solves the full selected horizon as one MILP model

The `"daily"` mode is recommended when using the restricted/free Gurobi license.

## Outputs

The project saves results in the `results/` folder:

* `gurobi_generation.csv`
* `ga_generation.csv`
* `ga_history.csv`
* `comparison.csv`
* total generation plots
* generation-by-plant plots
* GA convergence plot

## Modeling Notes

* Nuclear plants are forced to stay on.
* Nuclear plants have 80% minimum generation.
* Non-nuclear plants have 1% minimum generation when on.
* Ramp-up and ramp-down constraints are included in the Gurobi MILP.
* The first-hour ramp constraint is not imposed because the dataset does not contain the actual generation level before the selected scheduling horizon.
* The Genetic Algorithm uses a binary chromosome for plant on/off decisions.
* The GA uses a penalty function for unmet demand, oversupply, ramp violations, and nuclear-off violations.
* Health-cost data is available for July 2007, so the model reuses that day/hour pattern for July scheduling experiments.

## Methods Compared

| Method            | Description                                 |
| ----------------- | ------------------------------------------- |
| Gurobi MILP       | Exact mathematical optimization model       |
| Genetic Algorithm | Nature-inspired metaheuristic approximation |

The comparison is based on objective value, runtime, feasibility penalty, and percentage gap from the Gurobi solution.
