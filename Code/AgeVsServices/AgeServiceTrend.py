# AgeServiceTrend.py - Analyze and visualize trends in service usage by age group across multiple years of survey data.

from pathlib import Path
import re
import unicodedata
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR.parent
TARGET_YEARS = ["2022", "2023", "2024", "2025"]
TOP_N_SERVICES = 5
YEAR_FILES = {"2022": "22.csv", "2023": "23.csv", "2024": "24.csv", "2025": "25.csv"}

AGE_ORDER = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65-74",
    "75 or older",
]

AGE_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
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


def slugify(value: str) -> str:
    slug = normalize_text(value)
    slug = slug.replace("'", "")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def build_service_age_trend(records: pd.DataFrame, service_name: str) -> pd.DataFrame:
    service_records = records[records["service"] == service_name]
    if service_records.empty:
        return pd.DataFrame(index=TARGET_YEARS, columns=AGE_ORDER).fillna(0)
    return pd.crosstab(service_records["year"], service_records["age_group"]).reindex(
        index=TARGET_YEARS,
        columns=AGE_ORDER,
        fill_value=0,
    )


def plot_service_trend(trend: pd.DataFrame, service_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.2))

    for idx, age_group in enumerate(AGE_ORDER):
        ax.plot(
            trend.index.tolist(),
            trend[age_group].tolist(),
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            label=age_group,
            color=AGE_COLORS[idx],
        )

    ax.set_title(f"{service_name} by Age Group (2022-2025)", fontsize=16, color="black")
    ax.set_xlabel("Year", fontsize=12, color="black")
    ax.set_ylabel("Count", fontsize=12, color="black")
    ax.set_xticks(range(len(TARGET_YEARS)))
    ax.set_xticklabels(TARGET_YEARS)
    ax.tick_params(axis="both", colors="black")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax.legend(title="Age Group", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    fig.tight_layout()
    output_path = BASE_DIR / f"AgeServiceTrend_{slugify(service_name)}.png"
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def main() -> None:
    records = build_age_service_records()
    if records.empty:
        return

    top_services = records["service"].value_counts().head(TOP_N_SERVICES).index.tolist()

    for service_name in top_services:
        trend = build_service_age_trend(records, service_name)
        plot_service_trend(trend, service_name)

if __name__ == "__main__":
    main()
