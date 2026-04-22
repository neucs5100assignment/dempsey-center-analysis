# AgeTrend.py - Analyze and visualize age group trends across multiple years of survey data.

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75 or older"]
AGE_COLORS = ["#9ecae1", "#a1d99b", "#fddc8c", "#f4a6c1", "#bcb7ff", "#f5b38f", "#9dd9d2"]
YEAR_FILES = {"2022": "22.csv", "2023": "23.csv", "2024": "24.csv", "2025": "25.csv"}


def normalize_age(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in AGE_ORDER:
        return text
    if match := re.fullmatch(r"(\d{1,3})\s*-\s*(\d{1,3})", text):
        return f"{int(match.group(1))}-{int(match.group(2))}"
    if match := re.fullmatch(r"(\d{1,3})\s+or\s+older", text.lower()):
        return f"{int(match.group(1))} or older"
    if text.isdigit():
        return next((label for limit, label in zip([25, 35, 45, 55, 65, 75], AGE_ORDER) if int(text) < limit), "75 or older")
    return None


frames = []
for year, file_name in YEAR_FILES.items():
    df = pd.read_csv(BASE_DIR.parent / file_name)
    age_col = next((c for c in df.columns if str(c).lower() == "age"), None)
    if age_col is None:
        continue
    frames.append(pd.DataFrame({"year": year, "age_group": df[age_col].map(normalize_age)}))

age_df = pd.concat(frames, ignore_index=True).dropna()
table = pd.crosstab(age_df["year"], age_df["age_group"]).reindex(index=list(YEAR_FILES), columns=AGE_ORDER, fill_value=0)
table = table.div(table.sum(axis=1), axis=0).fillna(0) * 100

bottom = pd.Series(0, index=table.index)
plt.figure(figsize=(10, 6))
for group, color in zip(AGE_ORDER, AGE_COLORS):
    plt.bar(table.index, table[group], bottom=bottom, label=group, color=color, edgecolor="white", linewidth=0.6)
    bottom += table[group]

plt.title("Age Group Trend by Year (100% Stacked)")
plt.xlabel("Year")
plt.ylabel("Percent of Respondents")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(title="Age Group", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(BASE_DIR / "AgeTrend.png", dpi=300)
