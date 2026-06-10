from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class ProblemData:
    plants: List[str]
    time_periods: List[int]
    time_table: pd.DataFrame
    demand: Dict[int, float]
    plant_type: Dict[str, str]
    fuel_type: Dict[str, str]
    nuclear_plants: List[str]
    capacity: Dict[str, float]
    min_generation_fraction: Dict[str, float]
    ramp_fraction: Dict[str, float]
    fuel_cost: Dict[str, float]
    operating_cost: Dict[str, float]
    startup_cost: Dict[str, float]
    shutdown_cost: Dict[str, float]
    health_cost: Dict[Tuple[str, int], float]
    initial_on: Dict[str, int]


def _read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path)


def _cost_by_plant(cost_df: pd.DataFrame, year: int, fuel_type: Dict[str, str]) -> Dict[str, float]:
    if "year" not in cost_df.columns:
        # fixed_costs_revised.csv uses Unnamed: 0 for the year.
        if "Unnamed: 0" in cost_df.columns:
            cost_df = cost_df.rename(columns={"Unnamed: 0": "year"})
        else:
            raise ValueError("Cost table must contain a 'year' column.")

    row = cost_df[cost_df["year"] == year]
    if row.empty:
        available = sorted(cost_df["year"].unique().tolist())
        raise ValueError(f"Year {year} not found in cost table. Available years: {available}")

    row = row.iloc[0]
    return {plant: float(row[fuel]) for plant, fuel in fuel_type.items()}


def load_problem_data(
    data_dir: str | Path = "data",
    year: int = 2011,
    month: int = 7,
    start_day: int = 1,
    end_day: int = 7,
    health_cost_year: int = 2007,
) -> ProblemData:
    """Load and transform CSV files into dictionaries used by the optimization models.

    The demand data is selected for the chosen year/month/day interval.
    Health-cost data is available only for July 2007 in the uploaded files, so by default
    the model reuses the day/hour pattern from 2007 for the selected schedule.
    """
    data_dir = Path(data_dir)

    demand_df = _read_csv(data_dir, "demand.csv")
    plant_df = _read_csv(data_dir, "plant_capacities.csv")
    fuel_df = _read_csv(data_dir, "fuel_costs.csv")
    operating_df = _read_csv(data_dir, "operating_costs.csv")
    startup_df = _read_csv(data_dir, "startup_costs.csv")
    health_df = _read_csv(data_dir, "health_costs.csv")

    selected_demand = demand_df[
        (demand_df["YEAR"] == year)
        & (demand_df["MONTH"] == month)
        & (demand_df["DAY"].between(start_day, end_day))
    ].copy()

    if selected_demand.empty:
        raise ValueError(
            f"No demand rows found for year={year}, month={month}, "
            f"days={start_day}-{end_day}."
        )

    selected_demand = selected_demand.sort_values(["YEAR", "MONTH", "DAY", "HOUR"]).reset_index(drop=True)
    selected_demand["t"] = range(len(selected_demand))

    time_periods = selected_demand["t"].astype(int).tolist()
    time_table = selected_demand[["t", "YEAR", "MONTH", "DAY", "HOUR", "LOAD"]].copy()
    demand = dict(zip(selected_demand["t"].astype(int), selected_demand["LOAD"].astype(float)))

    plants = plant_df["Plant"].astype(str).tolist()
    plant_type = plant_df.set_index("Plant")["PlantType"].astype(str).to_dict()
    fuel_type = plant_df.set_index("Plant")["FuelType"].astype(str).to_dict()
    nuclear_plants = [p for p in plants if plant_type[p].upper() == "NUCLEAR"]
    capacity = plant_df.set_index("Plant")["Capacity"].astype(float).to_dict()

    min_generation_fraction = {p: 0.8 if p in nuclear_plants else 0.01 for p in plants}

    grouped_flexible = {"BIOMASS", "GAS", "HYDRO", "OIL"}
    ramp_fraction = {
        p: 1.0 if p in grouped_flexible else 0.2 if p in nuclear_plants else 0.25
        for p in plants
    }

    fuel_cost = _cost_by_plant(fuel_df, year, fuel_type)
    operating_cost = _cost_by_plant(operating_df, year, fuel_type)
    startup_cost = _cost_by_plant(startup_df, year, fuel_type)
    shutdown_cost = startup_cost.copy()

    # Map health costs by selected day/hour. Non-coal plants receive zero health cost.
    health_rows = health_df[
        (health_df["Year"] == health_cost_year)
        & (health_df["Month"] == month)
        & (health_df["Day"].between(start_day, end_day))
    ].copy()

    health_by_day_hour_plant = {
        (str(row.Plant), int(row.Day), int(row.Hour)): float(row.Cost)
        for row in health_rows.itertuples(index=False)
    }

    health_cost: Dict[Tuple[str, int], float] = {}
    for row in selected_demand.itertuples(index=False):
        # row order: t, YEAR, MONTH, DAY, HOUR, LOAD
        t = int(row.t)
        day = int(row.DAY)
        hour = int(row.HOUR)
        for plant in plants:
            health_cost[(plant, t)] = health_by_day_hour_plant.get((plant, day, hour), 0.0)

    # Assume nuclear units are initially on; all other plants initially off.
    # This makes startup/shutdown logic well-defined for t=0.
    initial_on = {p: 1 if p in nuclear_plants else 0 for p in plants}

    return ProblemData(
        plants=plants,
        time_periods=time_periods,
        time_table=time_table,
        demand=demand,
        plant_type=plant_type,
        fuel_type=fuel_type,
        nuclear_plants=nuclear_plants,
        capacity=capacity,
        min_generation_fraction=min_generation_fraction,
        ramp_fraction=ramp_fraction,
        fuel_cost=fuel_cost,
        operating_cost=operating_cost,
        startup_cost=startup_cost,
        shutdown_cost=shutdown_cost,
        health_cost=health_cost,
        initial_on=initial_on,
    )
