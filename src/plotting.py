from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_total_generation(generation_df: pd.DataFrame, title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = generation_df.groupby("t", as_index=False)["generation_mwh"].sum()
    plt.figure(figsize=(12, 5))
    plt.plot(total["t"], total["generation_mwh"], marker="o", markersize=2)
    plt.title(title)
    plt.xlabel("Time period")
    plt.ylabel("Total generation (MWh)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_generation_by_plant(generation_df: pd.DataFrame, title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = generation_df.pivot_table(index="t", columns="plant", values="generation_mwh", aggfunc="sum").fillna(0)
    plt.figure(figsize=(13, 6))
    for plant in pivot.columns:
        plt.plot(pivot.index, pivot[plant], label=plant)
    plt.title(title)
    plt.xlabel("Time period")
    plt.ylabel("Generation (MWh)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ga_history(history_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["generation"], history_df["best_fitness"])
    plt.title("Genetic Algorithm convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
