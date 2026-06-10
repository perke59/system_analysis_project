from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def summarize_results(gurobi_result: Dict[str, object] | None, ga_result: Dict[str, object]) -> pd.DataFrame:
    rows = []

    if gurobi_result is not None:
        rows.append(
            {
                "method": "Gurobi MILP",
                "status": gurobi_result.get("status"),
                "objective_or_cost": gurobi_result.get("objective"),
                "penalty": 0.0,
                "runtime_seconds": gurobi_result.get("runtime_seconds"),
                "mip_gap": gurobi_result.get("mip_gap"),
            }
        )

    rows.append(
        {
            "method": "Genetic Algorithm",
            "status": ga_result.get("status"),
            "objective_or_cost": ga_result.get("raw_cost"),
            "penalty": ga_result.get("penalty"),
            "runtime_seconds": ga_result.get("runtime_seconds"),
            "mip_gap": None,
        }
    )

    df = pd.DataFrame(rows)

    if gurobi_result is not None and gurobi_result.get("objective") is not None:
        gurobi_obj = float(gurobi_result["objective"])
        ga_cost = float(ga_result["raw_cost"])
        df["gap_vs_gurobi_percent"] = None
        df.loc[df["method"] == "Genetic Algorithm", "gap_vs_gurobi_percent"] = (
            (ga_cost - gurobi_obj) / gurobi_obj * 100.0
        )

    return df


def save_results(
    output_dir: str | Path,
    gurobi_result: Dict[str, object] | None,
    ga_result: Dict[str, object],
    comparison: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gurobi_result is not None and not gurobi_result.get("generation").empty:
        gurobi_result["generation"].to_csv(output_dir / "gurobi_generation.csv", index=False)

    ga_result["generation"].to_csv(output_dir / "ga_generation.csv", index=False)
    ga_result["history"].to_csv(output_dir / "ga_history.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
