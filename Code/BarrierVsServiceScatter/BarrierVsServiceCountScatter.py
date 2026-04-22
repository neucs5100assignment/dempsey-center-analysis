# BarrierVsServiceCountScatter.py - Create scatter plots showing the relationship between the number of barriers and the number of services
from pathlib import Path
import runpy
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_IMAGE = BASE_DIR / "BarrierVsServiceCountScatter.png"
TARGET_YEARS = ["2022", "2023", "2024", "2025"]

YEAR_FILE_NAMES = {
    "2022": "22.csv",
    "2023": "23.csv",
    "2024": "24.csv",
    "2025": "25.csv",
}


def is_selected(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def assign_columns_for_keywords(
    option_row: pd.Series,
    keywords: List[str],
    normalize_key: Callable[[str], str],
) -> List[Optional[str]]:
    grouped_cols: Dict[str, List[str]] = {}
    for col in option_row.index:
        key = normalize_key(option_row[col])
        grouped_cols.setdefault(key, []).append(col)

    used_by_keyword: Dict[str, int] = {}
    assigned_columns: List[Optional[str]] = []

    for keyword in keywords:
        norm_key = normalize_key(keyword)
        next_idx = used_by_keyword.get(norm_key, 0)
        matches = grouped_cols.get(norm_key, [])
        assigned = matches[next_idx] if next_idx < len(matches) else None
        assigned_columns.append(assigned)
        used_by_keyword[norm_key] = next_idx + 1

    return assigned_columns


def load_shared_defs() -> dict:
    heatmap_defs = runpy.run_path(str(BASE_DIR.parent / "AgeVsServices" / "AgeVsServicesHeatmap.py"))
    barrier_defs = runpy.run_path(str(BASE_DIR.parent / "Barrier" / "BarrierTrend.py"))

    return {
        "keywords_by_year": heatmap_defs["KEYWORDS_BY_YEAR"],
        "cleaned_service_label": heatmap_defs["cleaned_service_label"],
        "is_none_of_the_above": heatmap_defs["is_none_of_the_above"],
        "canonical_service_name": heatmap_defs["canonical_service_name"],
        "normalize_service": lambda v: heatmap_defs["normalize_text"](heatmap_defs["cleaned_service_label"](str(v or ""))),
        "standard_barriers": barrier_defs["STANDARD_BARRIERS"],
        "barrier_labels_by_year": barrier_defs["BARRIER_LABELS_BY_YEAR"],
        "normalize_barrier": barrier_defs["normalize_text"],
    }


def find_year_csv(year: str) -> Path:
    file_name = YEAR_FILE_NAMES[year]
    path = BASE_DIR.parent / file_name
    if not path.exists():
        raise FileNotFoundError(f"Expected file for year {year}: {path}")
    return path


def build_respondent_counts() -> pd.DataFrame:
    defs = load_shared_defs()
    records = []

    for year in TARGET_YEARS:
        df = pd.read_csv(find_year_csv(year))
        option_labels = pd.Series(df.columns, index=df.columns)

        barrier_keywords = [defs["barrier_labels_by_year"][year][name] for name in defs["standard_barriers"]]
        barrier_cols = assign_columns_for_keywords(option_labels, barrier_keywords, defs["normalize_barrier"])

        service_keywords = [k for k in defs["keywords_by_year"][year] if not defs["is_none_of_the_above"](k)]
        service_cols = assign_columns_for_keywords(option_labels, service_keywords, defs["normalize_service"])

        for row_idx in range(len(df)):
            barrier_count = sum(1 for col in barrier_cols if col is not None and is_selected(df.at[row_idx, col]))

            selected_services = set()
            for keyword, col in zip(service_keywords, service_cols):
                if col is None or not is_selected(df.at[row_idx, col]):
                    continue
                clean_label = defs["cleaned_service_label"](keyword)
                selected_services.add(defs["canonical_service_name"](clean_label))

            records.append(
                {
                    "year": year,
                    "barrier_count": barrier_count,
                    "service_count": len(selected_services),
                }
            )

    return pd.DataFrame(records)


def plot_scatter(df_counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    flat_axes = axes.flatten()

    colors = {
        "2022": "#1f77b4",
        "2023": "#ff7f0e",
        "2024": "#2ca02c",
        "2025": "#d62728",
    }

    x_ticks = range(0, int(df_counts["barrier_count"].max()) + 1)
    y_ticks = range(0, int(df_counts["service_count"].max()) + 1)

    for ax, year in zip(flat_axes, TARGET_YEARS):
        sub = df_counts[df_counts["year"] == year]
        if sub.empty:
            continue

        point_counts = sub.groupby(["barrier_count", "service_count"]).size().reset_index(name="n")

        bubble_sizes = point_counts["n"] * 22 + 18

        ax.scatter(
            point_counts["barrier_count"],
            point_counts["service_count"],
            s=bubble_sizes,
            alpha=0.45,
            color=colors[year],
            edgecolors="black",
            linewidths=0.4,
        )

        for _, row in point_counts.iterrows():
            ax.text(
                row["barrier_count"],
                row["service_count"],
                str(int(row["n"])),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

        ax.set_title(year, fontsize=13)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Barrier Count", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Service Count", fontsize=11)

    fig.suptitle("Barrier Count vs Service Count (4 Yearly Count-Bubble Plots)", fontsize=16)

    fig.savefig(OUTPUT_IMAGE, dpi=260)
    plt.close(fig)


def main() -> None:
    counts = build_respondent_counts()
    plot_scatter(counts)
    print(f"Generated: {OUTPUT_IMAGE.name}")
    print(f"Rows plotted: {len(counts)}")


if __name__ == "__main__":
    main()
