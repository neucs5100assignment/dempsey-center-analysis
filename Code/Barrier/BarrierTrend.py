# BarrierTrend.py - Analyze and visualize trends in common barriers to service access across multiple years of survey data.

from pathlib import Path
import re
import unicodedata
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_IMAGE = BASE_DIR / "BarrierTrend.png"
TARGET_YEARS = ["2022", "2023", "2024", "2025"]

YEAR_FILE_NAMES = {
    "2022": "22.csv",
    "2023": "23.csv",
    "2024": "24.csv",
    "2025": "25.csv",
}

STANDARD_BARRIERS = [
    "Times of services",
    "Physical location of services",
    "Transportation",
    "Reliable internet access or lack of reliable hardware",
]

BARRIER_LABELS_BY_YEAR = {
    "2022": {
        "Times of services": "Times of services",
        "Physical location of services": "Physical location of services",
        "Transportation": "Transportation",
        "Reliable internet access or lack of reliable hardware": "Reliable internet access or lack of reliable hardware",
    },
    "2023": {
        "Times of services": "Times of services",
        "Physical location of services": "Physical location of services",
        "Transportation": "Transportation",
        "Reliable internet access or lack of reliable hardware": "Reliable internet access or lack of reliable hardware",
    },
    "2024": {
        "Times of services": "Times of services (the service I was interested in wasn't scheduled at a time that worked for me)",
        "Physical location of services": "Physical location of services",
        "Transportation": "Transportation",
        "Reliable internet access or lack of reliable hardware": "Reliable internet access or lack of reliable hardware",
    },
    "2025": {
        "Times of services": "Times of services (the service I was interested in wasn't scheduled at a time that worked for me)",
        "Physical location of services": "Physical location of services",
        "Transportation": "Transportation",
        "Reliable internet access or lack of reliable hardware": "Reliable internet access or lack of reliable hardware",
    },
}


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def count_non_empty(series: pd.Series) -> int:
    text = series.astype(str).str.strip()
    return int(((~series.isna()) & (text != "") & (text.str.lower() != "nan")).sum())


def assign_columns_for_keywords(option_row: pd.Series, keywords: List[str]) -> List[Optional[str]]:
    grouped_cols: dict[str, list[str]] = {}
    for col in option_row.index:
        key = normalize_text(option_row[col])
        grouped_cols.setdefault(key, []).append(col)

    used_by_keyword: dict[str, int] = {}
    assigned_columns: List[Optional[str]] = []

    for keyword in keywords:
        norm_key = normalize_text(keyword)
        next_idx = used_by_keyword.get(norm_key, 0)
        matches = grouped_cols.get(norm_key, [])
        assigned = matches[next_idx] if next_idx < len(matches) else None
        assigned_columns.append(assigned)
        used_by_keyword[norm_key] = next_idx + 1

    return assigned_columns


def find_year_csv(year: str) -> Path:
    file_name = YEAR_FILE_NAMES[year]
    path = BASE_DIR.parent / file_name
    if not path.exists():
        raise FileNotFoundError(f"Expected file for year {year}: {path}")
    return path


def build_trend() -> pd.DataFrame:
    records = []
    for year in TARGET_YEARS:
        csv_path = find_year_csv(year)
        yearly_df = pd.read_csv(csv_path)

        keywords = [BARRIER_LABELS_BY_YEAR[year][name] for name in STANDARD_BARRIERS]
        header_labels = pd.Series(yearly_df.columns, index=yearly_df.columns)
        assigned_cols = assign_columns_for_keywords(header_labels, keywords)

        for barrier_name, col in zip(STANDARD_BARRIERS, assigned_cols):
            count = 0 if col is None else count_non_empty(yearly_df[col])
            records.append({"year": year, "barrier": barrier_name, "count": count})

    trend = pd.DataFrame(records).pivot(index="year", columns="barrier", values="count").reindex(
        TARGET_YEARS,
        fill_value=0,
    )

    existing_order = [c for c in STANDARD_BARRIERS if c in trend.columns]
    trend = trend[existing_order]
    return trend


def plot_trend(trend: pd.DataFrame) -> None:
    fig, (ax, ax_legend) = plt.subplots(
        ncols=2,
        figsize=(14.5, 6.8),
        gridspec_kw={"width_ratios": [4.5, 2.0]},
    )

    x_years = trend.index.tolist()
    for barrier_name in trend.columns:
        ax.plot(
            x_years,
            trend[barrier_name].tolist(),
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=barrier_name,
        )

    ax.set_title("Common Barrier Trend (2022-2025)", fontsize=14)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xticks(x_years)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.axis("off")
    ax_legend.legend(
        handles,
        labels,
        title="Barrier",
        fontsize=9,
        title_fontsize=10,
        loc="center left",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_IMAGE, dpi=220)
    plt.close()


def main() -> None:
    trend = build_trend()
    plot_trend(trend)
    print(f"Generated: {OUTPUT_IMAGE.name}")


if __name__ == "__main__":
    main()
