# AgeVsServicesHeatmap.py - Generate heatmaps showing the relationship between age groups and service usage for each year of survey data, focusing on the top 5 most used services per year.

from pathlib import Path
import re
import unicodedata
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR.parent
OUTPUT_IMAGE_TEMPLATE = BASE_DIR / "AgeVsServicesHeatmap_{year}.png"
TARGET_YEARS = ["2022", "2023", "2024", "2025"]
TOP_N_SERVICES = 5
YEAR_FILES = {"2022": "22.csv", "2023": "23.csv", "2024": "24.csv", "2025": "25.csv"}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "soft_orange",
    ["#fff8e6", "#fde8b2", "#f7c97d", "#f0a85b", "#de7f2d"],
)

AGE_ORDER = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65-74",
    "75 or older",
]

KEYWORDS_BY_YEAR = {
    "2022": [
        "Acupuncture",
        "Adult Counseling - Individual",
        "Adult Counseling - Support Groups",
        "Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, etc.)",
        "Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)",
        "Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)",
        "Dempsey Soup Program",
        "Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)",
        "Expressive Arts Workshop (e.g., Writing Through Cancer, Music for Wellness, etc.)",
        "Legacy & Life Reflections",
        "Massage Therapy",
        "Movement & Fitness Consult",
        "Movement & Fitness Classes/Workshops",
        "Nutrition Consult",
        "Nutrition Classes/Workshops",
        "Reiki Session",
        "Wig & Headwear Consult",
        "Youth & Family Counseling – Individual",
        "Youth & Family Counseling – Group/Support Group",
        "Youth & Family Classes/Workshops",
    ],
    "2023": [
        "Acupuncture",
        "Clayton’s House stay",
        "Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, etc.)",
        "Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)",
        "Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)",
        "Dempsey Soup Program",
        "Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)",
        "Expressive Arts Workshop (e.g., Writing Through Cancer, Music for Wellness, etc.)",
        "Individual Counseling",
        "Legacy & Life Reflections",
        "Massage Therapy",
        "Movement & Fitness Classes/Workshops",
        "Movement & Fitness Consult",
        "Nutrition Consult",
        "Nutrition Classes/Workshops",
        "Reiki Session",
        "Support Groups",
        "Wig & Headwear Consult",
        "Youth & Family Counseling",
        "Youth & Family Classes/Workshops",
    ],
    "2024": [
        "Clayton’s House Stay",
        "Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, mastectomy supplies, etc.)",
        "Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)",
        "Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)",
        "Expressive Arts Workshop (e.g., Writing Through Cancer, Music for Wellness, etc.)",
        "Wig & Headwear Consult",
        "None of the above",
        "Acupuncture",
        "Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)",
        "Dempsey Soup Program",
        "Massage Therapy",
        "Movement & Fitness Consult",
        "Movement & Fitness Classes/Workshops",
        "Nutrition Consult",
        "Nutrition Classes/Workshops",
        "Reiki Session",
        "None of the above",
        "Individual Counseling",
        "Legacy & Life Reflections",
        "Support Groups",
        "Youth & Family Counseling/Consults",
        "Youth & Family Classes/Workshops",
    ],
    "2025": [
        "None of the above",
        "Clayton’s House Stay (no-cost lodging near Portland, ME for clients traveling more than 30 miles for treatment)",
        "Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, mastectomy supplies, etc.)",
        "Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)",
        "Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)",
        "Expressive Arts Workshop (e.g., Writing Through Cancer, Music for Wellness, etc.)",
        "Wig & Headwear Consult (personalized guidance and fittings from professionals trained to support individuals experienceing hair loss due to cancer or its treatment)",
        "None of the above",
        "Acupuncture (provided by licensed practitioners to help manage cancer-related symptoms such as pain, fatigue, and nausea)",
        "Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)",
        "Dempsey Soup Program (nutritious soups prepared by Dempsey Center nutrition staff and volunteers, distributed at no cost to clients)",
        "Massage Therapy (provided by licensed massage therapists trained to support individuals affected by cancer through gentle, therapeutic techniques)",
        "Movement & Fitness Consult (individual sessions with oncology-trained professionals to create safe, personalized movement and fitness plans)",
        "Movement & Fitness Classes/Workshops (e.g., Chair Yoga, Tai Chi; group classes led by instructors with oncology-specific training, designed to promote strength, mobility, and well-being in a supportive environment)",
        "Nutrition Consult (individual consultations with licensed and oncology-trained nutrition professionals to address nutrition concerns and provide personalized recommendations)",
        "Nutrition Classes/Workshops (e.g., What to Eat During Cancer Treatment; education and cooking classes led by oncology-trained nutrition staff, focused on dietary strategies to support health during and after cancer treatment)",
        "Reiki Session (provided by certified practitioners trained to support individuals affected by cancer through gentle, energy-based techniques that promote relaxation and balance)",
        "None of the above",
        "Individual Counseling (one-on-one sessions with licensed counselors specially trained to support individuals and families navigating the impact of cancer)",
        "Legacy & Life Reflections (recorded legacy interviews for individuals with advanced cancer to reflect on their lives and stories)",
        "Support Groups (professionally facilitated groups for sharing experiences, reducing isolation, and building connection)",
        "Youth & Family Parent/Guardian Consults (individual consultations for parents or guardians whose family has been impacted by cancer)",
        "Youth & Family Classes/Workshops (e.g., Family Nights, Space to Breathe therapeutic summer camp; programming designed for children, teens, and families)",
    ],
}

SERVICE_ALIAS_MAP = {
    "adult counseling - individual": "Individual Counseling",
    "adult counseling - support groups": "Support Groups",
    "youth & family counseling - individual": "Youth & Family Counseling",
    "youth & family counseling - group/support group": "Youth & Family Counseling",
    "youth & family counseling": "Youth & Family Counseling",
    "youth & family counseling/consults": "Youth & Family Counseling",
    "youth & family parent/guardian consults": "Youth & Family Counseling",
    "clayton's house stay": "Clayton's House Stay",
}


def normalize_age_group(age_value):
    if pd.isna(age_value):
        return None

    text = str(age_value).strip()
    if not text or text.lower() == "response":
        return None

    range_match = re.match(r"^(\d{1,3})\s*-\s*(\d{1,3})$", text)
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        return f"{low}-{high}"

    older_match = re.match(r"^(\d{1,3})\s+or\s+older$", text.lower())
    if older_match:
        return f"{older_match.group(1)} or older"

    if text.isdigit():
        value = int(text)
        bins = [25, 35, 45, 55, 65, 75]
        labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74"]
        for upper, label in zip(bins, labels):
            if value < upper:
                return label
        return "75 or older"

    return None


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def strip_parentheses_text(value: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cleaned_service_label(value: str) -> str:
    return strip_parentheses_text(str(value or ""))


def is_none_of_the_above(value: str) -> bool:
    return normalize_text(cleaned_service_label(value)) == "none of the above"


def is_selected(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def assign_columns_for_keywords(option_row: pd.Series, keywords: List[str]) -> List[Optional[str]]:
    grouped_cols: dict[str, list[str]] = {}
    for col in option_row.index:
        key = normalize_text(cleaned_service_label(option_row[col]))
        grouped_cols.setdefault(key, []).append(col)

    used_by_keyword: dict[str, int] = {}
    assigned_columns: List[Optional[str]] = []

    for keyword in keywords:
        norm_key = normalize_text(cleaned_service_label(keyword))
        next_idx = used_by_keyword.get(norm_key, 0)
        matches = grouped_cols.get(norm_key, [])
        assigned = matches[next_idx] if next_idx < len(matches) else None
        assigned_columns.append(assigned)
        used_by_keyword[norm_key] = next_idx + 1

    return assigned_columns


def canonical_service_name(raw_name: str) -> str:
    cleaned = strip_parentheses_text(raw_name)
    norm = normalize_text(cleaned)
    return SERVICE_ALIAS_MAP.get(norm, cleaned)


def build_age_service_records() -> pd.DataFrame:
    records = []

    for year in TARGET_YEARS:
        csv_path = CSV_DIR / YEAR_FILES[year]
        df = pd.read_csv(csv_path)

        option_labels = pd.Series(df.columns, index=df.columns)
        data = df.reset_index(drop=True)

        if "Age" not in df.columns:
            continue

        filtered_keywords = [k for k in KEYWORDS_BY_YEAR[year] if not is_none_of_the_above(k)]
        assigned_cols = assign_columns_for_keywords(option_labels, filtered_keywords)

        age_series = data["Age"].apply(normalize_age_group)

        for row_idx, age_group in age_series.items():
            if age_group is None:
                continue

            for keyword, col in zip(filtered_keywords, assigned_cols):
                if col is None:
                    continue
                if not is_selected(data.at[row_idx, col]):
                    continue

                service_name = canonical_service_name(cleaned_service_label(keyword))
                records.append(
                    {
                        "year": year,
                        "age_group": age_group,
                        "service": service_name,
                    }
                )

    return pd.DataFrame(records)


def build_heatmap_table(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(index=AGE_ORDER)

    top_services = records["service"].value_counts().head(TOP_N_SERVICES).index.tolist()

    filtered = records[records["service"].isin(top_services)]
    table = pd.crosstab(filtered["age_group"], filtered["service"])
    table = table.reindex(index=AGE_ORDER, fill_value=0)

    ordered_cols = [c for c in top_services if c in table.columns]
    return table[ordered_cols]


def plot_heatmap_for_year(records: pd.DataFrame, year: str) -> None:
    if records.empty:
        return

    pct_table = build_heatmap_table(records)

    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    last_im = ax.imshow(pct_table.values, cmap=HEATMAP_CMAP, aspect="auto")
    ax.set_title(f"{year} - Top 5 Services", fontsize=15, color="black")
    ax.set_xlabel("Service", fontsize=11, color="black")
    ax.set_ylabel("Age Group", fontsize=11, color="black")

    ax.set_xticks(range(len(pct_table.columns)))
    ax.set_xticklabels(pct_table.columns, rotation=35, ha="right", fontsize=9, color="black")
    ax.set_yticks(range(len(pct_table.index)))
    ax.set_yticklabels(pct_table.index, fontsize=10, color="black")
    ax.tick_params(axis="both", colors="black")

    for i in range(pct_table.shape[0]):
        for j in range(pct_table.shape[1]):
            val = pct_table.iat[i, j]
            if val > 0:
                ax.text(j, i, f"{int(val)}", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(last_im, ax=ax, shrink=0.9)
    cbar.set_label("Count", fontsize=11, color="black")
    cbar.ax.tick_params(colors="black")

    fig.suptitle(f"Age vs Services Heatmap - {year} (Top 5 Services, Counts)", fontsize=17, color="black")
    output_image = OUTPUT_IMAGE_TEMPLATE.with_name(OUTPUT_IMAGE_TEMPLATE.name.format(year=year))
    fig.tight_layout()
    fig.savefig(output_image, dpi=260)
    plt.close(fig)


def main() -> None:
    records = build_age_service_records()
    for year in TARGET_YEARS:
        year_records: pd.DataFrame = records.loc[records["year"].eq(year), :].copy()
        if year_records.empty:
            continue
        plot_heatmap_for_year(year_records, year)
    print("Generated 4 yearly heatmap PNGs")


if __name__ == "__main__":
    main()
