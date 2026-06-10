from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, Tuple

import pandas as pd

try:
    from .data_loader import ProblemData
except ImportError:  
    from data_loader import ProblemData


def solve_with_gurobi(data: ProblemData, time_limit: int | None = None, mip_gap: float | None = None, output_flag: int = 1):
    """Solve the unit commitment model exactly/as a MILP using Gurobi."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError(
            "gurobipy is not installed. Install it with: pip install gurobipy"
        ) from exc

    P = data.plants
    T = data.time_periods

    model = gp.Model("power_generation_unit_commitment")
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    z = model.addVars(P, T, lb=0.0, vtype=GRB.CONTINUOUS, name="generation")
    u = model.addVars(P, T, vtype=GRB.BINARY, name="on")
    v = model.addVars(P, T, vtype=GRB.BINARY, name="startup")
    w = model.addVars(P, T, vtype=GRB.BINARY, name="shutdown")

    objective = gp.quicksum(
        (data.fuel_cost[p] + data.health_cost[(p, t)]) * z[p, t]
        + data.operating_cost[p] * u[p, t]
        + data.startup_cost[p] * v[p, t]
        + data.shutdown_cost[p] * w[p, t]
        for p in P
        for t in T
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Demand must be exactly satisfied every hour.
    model.addConstrs(
        (gp.quicksum(z[p, t] for p in P) == data.demand[t] for t in T),
        name="meet_demand",
    )

    # Minimum and maximum generation conditional on plant on/off state.
    model.addConstrs(
        (z[p, t] >= data.min_generation_fraction[p] * data.capacity[p] * u[p, t] for p in P for t in T),
        name="min_generation_if_on",
    )
    model.addConstrs(
        (z[p, t] <= data.capacity[p] * u[p, t] for p in P for t in T),
        name="max_generation_if_on",
    )

    # Nuclear plants are always on.
    model.addConstrs(
        (u[p, t] == 1 for p in data.nuclear_plants for t in T),
        name="nuclear_always_on",
    )


    first_t = T[0]

    for prev_t, current_t in zip(T[:-1], T[1:]):
        model.addConstrs(
            (z[p, current_t] - z[p, prev_t] <= data.ramp_fraction[p] * data.capacity[p] for p in P),
            name=f"ramp_up[{current_t}]",
        )
        model.addConstrs(
            (z[p, current_t] - z[p, prev_t] >= -data.ramp_fraction[p] * data.capacity[p] for p in P),
            name=f"ramp_down[{current_t}]",
        )

    # Startup/shutdown consistency.
    model.addConstrs((v[p, t] <= u[p, t] for p in P for t in T), name="startup_implies_on")
    model.addConstrs((w[p, t] <= 1 - u[p, t] for p in P for t in T), name="shutdown_implies_off")

    for p in P:
        model.addConstr(
            v[p, first_t] - w[p, first_t] == u[p, first_t] - data.initial_on[p],
            name=f"initial_start_stop_link[{p}]",
        )
    for prev_t, current_t in zip(T[:-1], T[1:]):
        model.addConstrs(
            (v[p, current_t] - w[p, current_t] == u[p, current_t] - u[p, prev_t] for p in P),
            name=f"start_stop_link[{current_t}]",
        )

    start = time.perf_counter()
    try:
        model.optimize()
    except Exception as exc:
        runtime = time.perf_counter() - start
        return {
            "status": "SKIPPED",
            "objective": None,
            "runtime_seconds": runtime,
            "generation": pd.DataFrame(),
            "on_off": pd.DataFrame(),
            "model": model,
            "message": (
                "Gurobi could not solve this model. This usually happens with the "
                "free/restricted size-limited license. Use GUROBI_MODE='daily' or reduce "
                "the number of days, or use an academic/unrestricted Gurobi license. "
                f"Original error: {exc}"
            ),
        }
    runtime = time.perf_counter() - start

    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }.get(model.Status, str(model.Status))

    if model.SolCount == 0:
        return {
            "status": status_name,
            "objective": None,
            "runtime_seconds": runtime,
            "generation": pd.DataFrame(),
            "on_off": pd.DataFrame(),
            "model": model,
            "message": None,
        }

    rows = []
    on_rows = []
    time_lookup = data.time_table.set_index("t").to_dict(orient="index")
    for t in T:
        info = time_lookup[t]
        for p in P:
            rows.append(
                {
                    "t": t,
                    "year": info["YEAR"],
                    "month": info["MONTH"],
                    "day": info["DAY"],
                    "hour": info["HOUR"],
                    "plant": p,
                    "generation_mwh": z[p, t].X,
                    "on": round(u[p, t].X),
                    "startup": round(v[p, t].X),
                    "shutdown": round(w[p, t].X),
                }
            )
            on_rows.append({"t": t, "plant": p, "on": round(u[p, t].X)})

    return {
        "status": status_name,
        "objective": float(model.ObjVal),
        "best_bound": float(model.ObjBound) if hasattr(model, "ObjBound") else None,
        "mip_gap": float(model.MIPGap) if hasattr(model, "MIPGap") else None,
        "runtime_seconds": runtime,
        "generation": pd.DataFrame(rows),
        "on_off": pd.DataFrame(on_rows),
        "model": model,
        "message": None,
    }



def _subset_problem_data(data: ProblemData, selected_t: list[int]) -> ProblemData:
    """Create a smaller ProblemData object for a subset of time periods."""
    selected_set = set(selected_t)
    return replace(
        data,
        time_periods=selected_t,
        time_table=data.time_table[data.time_table["t"].isin(selected_set)].copy(),
        demand={t: data.demand[t] for t in selected_t},
        health_cost={(p, t): c for (p, t), c in data.health_cost.items() if t in selected_set},
    )


def solve_with_gurobi_by_day(
    data: ProblemData,
    time_limit_per_day: int | None = 120,
    mip_gap: float | None = 0.01,
):
    """Solve each day as a separate 24-hour MILP and combine the results.

    This is a rolling-horizon workaround for Gurobi's restricted/free license size limit.
    It is not exactly the same as solving the whole week in one MILP, because the days are
    optimized separately, but it is valid for comparing exact MILP daily schedules with GA.
    """
    total_runtime = 0.0
    total_objective = 0.0
    generation_frames = []
    on_off_frames = []
    statuses = []
    messages = []

    grouped = data.time_table.groupby(["YEAR", "MONTH", "DAY"], sort=True)
    for (year, month, day), day_table in grouped:
        selected_t = day_table["t"].astype(int).tolist()
        day_data = _subset_problem_data(data, selected_t)
        print(f"  Solving Gurobi day {int(day)} with {len(selected_t)} hours...")
        result = solve_with_gurobi(
            day_data,
            time_limit=time_limit_per_day,
            mip_gap=mip_gap,
            output_flag=0,
        )
        statuses.append(f"day {int(day)}: {result['status']}")
        total_runtime += float(result.get("runtime_seconds", 0.0))
        if result.get("message"):
            messages.append(f"day {int(day)}: {result['message']}")
        if result.get("objective") is not None:
            total_objective += float(result["objective"])
        if result.get("generation") is not None and not result["generation"].empty:
            generation_frames.append(result["generation"])
        if result.get("on_off") is not None and not result["on_off"].empty:
            on_off_frames.append(result["on_off"])

    generation = pd.concat(generation_frames, ignore_index=True) if generation_frames else pd.DataFrame()
    on_off = pd.concat(on_off_frames, ignore_index=True) if on_off_frames else pd.DataFrame()
    ok = bool(generation_frames) and all("OPTIMAL" in s or "TIME_LIMIT" in s for s in statuses)

    return {
        "status": "DAILY_ROLLING_OK" if ok else "DAILY_ROLLING_PARTIAL",
        "objective": total_objective if generation_frames else None,
        "runtime_seconds": total_runtime,
        "generation": generation,
        "on_off": on_off,
        "model": None,
        "message": "; ".join(messages) if messages else "Daily rolling horizon was used to avoid the restricted-license model-size limit.",
        "daily_statuses": statuses,
    }


if __name__ == "__main__":
    print("This file contains solver functions. Run the full project with: python main.py")
