from __future__ import annotations

from pathlib import Path

from src.data_loader import load_problem_data
from src.evaluation import save_results, summarize_results
from src.genetic_algorithm import GAConfig, GeneticAlgorithmSolver
from src.plotting import plot_ga_history, plot_generation_by_plant, plot_total_generation

# had trouble with paths  
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


YEAR = 2011
MONTH = 7
START_DAY = 1
END_DAY = 1

RUN_GUROBI = True
RUN_GA = True


GUROBI_MODE = "daily"   # options: "daily", "combined"


def _plot_if_available(result: dict, prefix: str, title: str) -> None:
    if result is None or result.get("generation") is None or result["generation"].empty:
        return
    plot_total_generation(
        result["generation"],
        f"{title} total generation",
        RESULTS_DIR / f"{prefix}_total_generation.png",
    )
    plot_generation_by_plant(
        result["generation"],
        f"{title} generation by plant",
        RESULTS_DIR / f"{prefix}_generation_by_plant.png",
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    data = load_problem_data(
        data_dir=DATA_DIR,
        year=YEAR,
        month=MONTH,
        start_day=START_DAY,
        end_day=END_DAY,
    )

    print(f"Plants: {len(data.plants)}")
    print(f"Time periods: {len(data.time_periods)}")
    print(f"Total demand: {sum(data.demand.values()):,.2f} MWh")

    gurobi_result = None
    if RUN_GUROBI:
        from src.gurobi_model import solve_with_gurobi, solve_with_gurobi_by_day

        print("\nSolving with Gurobi MILP...")
        if GUROBI_MODE.lower() == "daily":
            print("Gurobi mode: daily rolling horizon. This avoids the free-license model-size limit.")
            gurobi_result = solve_with_gurobi_by_day(data, time_limit_per_day=120, mip_gap=0.01)
        else:
            print("Gurobi mode: combined full-horizon MILP.")
            gurobi_result = solve_with_gurobi(data, time_limit=300, mip_gap=0.01)

        print(f"Gurobi status: {gurobi_result['status']}")
        print(f"Gurobi objective: {gurobi_result['objective']}")
        print(f"Gurobi runtime: {gurobi_result['runtime_seconds']:.2f} seconds")
        if gurobi_result.get("message"):
            print(f"Gurobi message: {gurobi_result['message']}")

        _plot_if_available(gurobi_result, "gurobi", "Gurobi")

    ga_result = None
    if RUN_GA:
        print("\nSolving with Genetic Algorithm...")
        config = GAConfig(
            population_size=50,
            generations=80,
            crossover_rate=0.85,
            mutation_rate=0.015,
            random_seed=42,
        )
        ga_solver = GeneticAlgorithmSolver(data, config)
        ga_result = ga_solver.solve()
        print(f"GA raw cost: {ga_result['raw_cost']}")
        print(f"GA penalty: {ga_result['penalty']}")
        print(f"GA fitness: {ga_result['fitness']}")
        print(f"GA runtime: {ga_result['runtime_seconds']:.2f} seconds")

        _plot_if_available(ga_result, "ga", "Genetic Algorithm")
        plot_ga_history(ga_result["history"], RESULTS_DIR / "ga_convergence.png")

    if ga_result is None:
        raise RuntimeError("GA result is missing. Set RUN_GA=True.")

    comparison = summarize_results(gurobi_result, ga_result)
    save_results(RESULTS_DIR, gurobi_result, ga_result, comparison)

    print("\nComparison:")
    print(comparison.to_string(index=False))
    print(f"\nSaved results to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
