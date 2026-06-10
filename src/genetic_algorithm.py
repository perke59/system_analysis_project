from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .data_loader import ProblemData
except ImportError:  # Allows running this file directly in Visual Studio.
    from data_loader import ProblemData


@dataclass
class GAConfig:
    population_size: int = 80
    generations: int = 150
    crossover_rate: float = 0.85
    mutation_rate: float = 0.015
    tournament_size: int = 3
    elitism: int = 2
    random_seed: int = 42
    penalty_unmet_demand: float = 1_000_000.0
    penalty_oversupply: float = 250_000.0
    penalty_ramp: float = 100_000.0
    penalty_nuclear_off: float = 1_000_000.0


@dataclass
class FitnessResult:
    cost: float
    raw_cost: float
    penalty: float
    generation: np.ndarray
    demand_gap: np.ndarray
    ramp_violation: float


class GeneticAlgorithmSolver:
    """Binary Genetic Algorithm for the on/off part of unit commitment.

    Chromosome shape: (number_of_plants, number_of_time_periods)
    Gene value 1 means the plant is ON, 0 means OFF.

    A greedy economic dispatch is used to assign generation to active plants.
    Constraint violations are added to the fitness as penalties.
    """

    def __init__(self, data: ProblemData, config: GAConfig | None = None):
        self.data = data
        self.config = config or GAConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        random.seed(self.config.random_seed)

        self.P = data.plants
        self.T = data.time_periods
        self.n_plants = len(self.P)
        self.n_times = len(self.T)
        self.plant_index = {p: i for i, p in enumerate(self.P)}

        # Dispatch priority by variable cost; cheaper plants are filled first.
        avg_variable_cost = []
        for p in self.P:
            avg_health = np.mean([data.health_cost[(p, t)] for t in self.T])
            avg_variable_cost.append((data.fuel_cost[p] + avg_health, p))
        self.dispatch_order = [p for _, p in sorted(avg_variable_cost)]

    def _new_individual(self) -> np.ndarray:
        individual = self.rng.integers(0, 2, size=(self.n_plants, self.n_times), dtype=np.int8)

        # Nuclear plants must be on.
        for p in self.data.nuclear_plants:
            individual[self.plant_index[p], :] = 1

        # Repair obvious infeasibility: ensure enough active capacity for each hour.
        for tj, t in enumerate(self.T):
            while self._active_capacity(individual[:, tj]) < self.data.demand[t]:
                off_candidates = [i for i in range(self.n_plants) if individual[i, tj] == 0]
                if not off_candidates:
                    break
                # Turn on the cheapest currently-off plant.
                cheapest = min(off_candidates, key=lambda idx: self.data.fuel_cost[self.P[idx]])
                individual[cheapest, tj] = 1
        return individual

    def _active_capacity(self, on_vector: np.ndarray) -> float:
        return sum(self.data.capacity[p] for p, gene in zip(self.P, on_vector) if gene == 1)

    def _dispatch_for_individual(self, individual: np.ndarray) -> FitnessResult:
        generation = np.zeros((self.n_plants, self.n_times), dtype=float)
        penalty = 0.0
        raw_cost = 0.0
        demand_gap = np.zeros(self.n_times, dtype=float)
        ramp_violation = 0.0

        # No ramp penalty is applied before the first selected hour because the dataset
        # does not include the actual generation immediately before the schedule starts.
        # Applying a ramp-from-zero assumption here makes the first hour unrealistically hard.
        previous_generation = None
        previous_on = np.array([self.data.initial_on[p] for p in self.P], dtype=np.int8)

        for tj, t in enumerate(self.T):
            on = individual[:, tj].copy()

            for p in self.data.nuclear_plants:
                on[self.plant_index[p]] = 1

            min_gen = np.zeros(self.n_plants)
            max_gen = np.zeros(self.n_plants)
            for i, p in enumerate(self.P):
                if on[i] == 1:
                    min_gen[i] = self.data.min_generation_fraction[p] * self.data.capacity[p]
                    max_gen[i] = self.data.capacity[p]

            current_generation = min_gen.copy()
            remaining = self.data.demand[t] - current_generation.sum()

            # If minimum generation already exceeds demand, keep minimum generation and penalize oversupply.
            if remaining < 0:
                penalty += abs(remaining) * self.config.penalty_oversupply
                demand_gap[tj] = remaining
            else:
                # Fill remaining demand using cheapest active plants.
                for p in self.dispatch_order:
                    i = self.plant_index[p]
                    if on[i] == 0:
                        continue
                    available = max_gen[i] - current_generation[i]
                    add = min(available, remaining)
                    current_generation[i] += add
                    remaining -= add
                    if remaining <= 1e-6:
                        break

                if remaining > 1e-6:
                    penalty += remaining * self.config.penalty_unmet_demand
                    demand_gap[tj] = remaining

            # Ramp penalty. Skip it for the first selected hour because previous
            # generation is unknown. Apply it only between modeled consecutive hours.
            if previous_generation is not None:
                for i, p in enumerate(self.P):
                    ramp_limit = self.data.ramp_fraction[p] * self.data.capacity[p]
                    diff = abs(current_generation[i] - previous_generation[i])
                    if diff > ramp_limit + 1e-6:
                        excess = diff - ramp_limit
                        ramp_violation += excess
                        penalty += excess * self.config.penalty_ramp

            # Nuclear off penalty, mainly protective because we force repair.
            for p in self.data.nuclear_plants:
                i = self.plant_index[p]
                if individual[i, tj] == 0:
                    penalty += self.config.penalty_nuclear_off

            # Cost for this hour.
            for i, p in enumerate(self.P):
                startup = 1 if on[i] == 1 and previous_on[i] == 0 else 0
                shutdown = 1 if on[i] == 0 and previous_on[i] == 1 else 0
                raw_cost += (self.data.fuel_cost[p] + self.data.health_cost[(p, t)]) * current_generation[i]
                raw_cost += self.data.operating_cost[p] * on[i]
                raw_cost += self.data.startup_cost[p] * startup
                raw_cost += self.data.shutdown_cost[p] * shutdown

            generation[:, tj] = current_generation
            previous_generation = current_generation
            previous_on = on

        return FitnessResult(
            cost=raw_cost + penalty,
            raw_cost=raw_cost,
            penalty=penalty,
            generation=generation,
            demand_gap=demand_gap,
            ramp_violation=ramp_violation,
        )

    def _tournament_select(self, population: List[np.ndarray], fitness_values: List[float]) -> np.ndarray:
        candidates = self.rng.choice(len(population), size=self.config.tournament_size, replace=False)
        best_idx = min(candidates, key=lambda idx: fitness_values[idx])
        return population[best_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Time-based one-point crossover preserves each plant schedule blocks.
        cut = int(self.rng.integers(1, self.n_times))
        child1 = np.concatenate([parent1[:, :cut], parent2[:, cut:]], axis=1)
        child2 = np.concatenate([parent2[:, :cut], parent1[:, cut:]], axis=1)
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutation_mask = self.rng.random(size=individual.shape) < self.config.mutation_rate
        individual = np.where(mutation_mask, 1 - individual, individual).astype(np.int8)

        # Nuclear plants must stay on.
        for p in self.data.nuclear_plants:
            individual[self.plant_index[p], :] = 1
        return individual

    def solve(self) -> Dict[str, object]:
        start = time.perf_counter()
        population = [self._new_individual() for _ in range(self.config.population_size)]

        best_individual = None
        best_result = None
        history = []

        for generation_id in range(self.config.generations):
            results = [self._dispatch_for_individual(ind) for ind in population]
            fitness_values = [res.cost for res in results]

            best_idx = int(np.argmin(fitness_values))
            generation_best = results[best_idx]
            if best_result is None or generation_best.cost < best_result.cost:
                best_result = generation_best
                best_individual = population[best_idx].copy()

            history.append(
                {
                    "generation": generation_id,
                    "best_fitness": best_result.cost,
                    "best_raw_cost": best_result.raw_cost,
                    "best_penalty": best_result.penalty,
                    "average_fitness": float(np.mean(fitness_values)),
                }
            )

            # Elitism.
            elite_indices = np.argsort(fitness_values)[: self.config.elitism]
            new_population = [population[idx].copy() for idx in elite_indices]

            while len(new_population) < self.config.population_size:
                p1 = self._tournament_select(population, fitness_values)
                p2 = self._tournament_select(population, fitness_values)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.config.population_size:
                    new_population.append(c2)

            population = new_population

        runtime = time.perf_counter() - start
        assert best_individual is not None and best_result is not None

        rows = []
        time_lookup = self.data.time_table.set_index("t").to_dict(orient="index")
        for tj, t in enumerate(self.T):
            info = time_lookup[t]
            for i, p in enumerate(self.P):
                rows.append(
                    {
                        "t": t,
                        "year": info["YEAR"],
                        "month": info["MONTH"],
                        "day": info["DAY"],
                        "hour": info["HOUR"],
                        "plant": p,
                        "generation_mwh": best_result.generation[i, tj],
                        "on": int(best_individual[i, tj]),
                    }
                )

        return {
            "status": "COMPLETED",
            "fitness": float(best_result.cost),
            "raw_cost": float(best_result.raw_cost),
            "penalty": float(best_result.penalty),
            "runtime_seconds": runtime,
            "ramp_violation": float(best_result.ramp_violation),
            "generation": pd.DataFrame(rows),
            "history": pd.DataFrame(history),
            "best_individual": best_individual,
        }


if __name__ == "__main__":
    print("This file contains the GeneticAlgorithmSolver class. Run the full project with: python main.py")
