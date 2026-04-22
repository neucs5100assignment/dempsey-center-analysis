from pathlib import Path
import re
import unicodedata
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_IMAGE = BASE_DIR / "ServiceTrend.png"
TARGET_YEARS = ["2022", "2023", "2024", "2025"]

YEAR_FILE_NAMES = {
    "2022": "22.csv",
    "2023": "23.csv",
    "2024": "24.csv",
    "2025": "25.csv",
}


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
    "clayton's house stay": "Clayton's House Stay",
}


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


def count_non_empty(series: pd.Series) -> int:
    text = series.astype(str).str.strip()
    return int(((~series.isna()) & (text != "") & (text.str.lower() != "nan")).sum())


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


def find_year_csv(year: str) -> Path:
    file_name = YEAR_FILE_NAMES[year]
    path = BASE_DIR.parent / file_name
    if not path.exists():
        raise FileNotFoundError(f"Expected file for year {year}: {path}")
    return path


def collect_service_counts() -> pd.DataFrame:
    records = []

    for year, keywords in KEYWORDS_BY_YEAR.items():
        df = pd.read_csv(find_year_csv(year))
        option_labels = pd.Series(df.columns, index=df.columns)

        filtered_keywords = [k for k in keywords if not is_none_of_the_above(k)]
        assigned_cols = assign_columns_for_keywords(option_labels, filtered_keywords)

        for keyword, col in zip(filtered_keywords, assigned_cols):
            cleaned_label = cleaned_service_label(keyword)
            count = 0 if col is None else count_non_empty(df[col])
            records.append({"year": year, "service": cleaned_label, "count": count})

    return pd.DataFrame(records)


def build_top10_count_trend() -> pd.DataFrame:
    raw_counts = collect_service_counts()
    raw_counts["year"] = raw_counts["year"].astype(str).str.strip()
    raw_counts["service_canonical"] = raw_counts["service"].apply(canonical_service_name)

    grouped = raw_counts.groupby(["service_canonical", "year"], as_index=False)["count"].sum()

    years_by_service: dict[str, set[str]] = {}
    for _, row in grouped.iterrows():
        service_name = str(row["service_canonical"])
        year_name = str(row["year"])
        years_by_service.setdefault(service_name, set()).add(year_name)

    services_in_all_years = [
        service_name
        for service_name, years in years_by_service.items()
        if len(years) == len(TARGET_YEARS)
    ]
    eligible = grouped[grouped["service_canonical"].isin(services_in_all_years)].copy()

    top_services = (
        eligible.groupby("service_canonical")["count"].sum().nlargest(10).index.to_list()
    )

    trend: pd.DataFrame = (
        eligible[eligible["service_canonical"].isin(top_services)]
        .pivot_table(
            index="year",
            columns="service_canonical",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(TARGET_YEARS, fill_value=0)
        .reindex(columns=top_services, fill_value=0)
    )

    return trend


def plot_trend(trend: pd.DataFrame) -> None:
    fig, (ax, ax_legend) = plt.subplots(
        ncols=2,
        figsize=(17.5, 9),
        gridspec_kw={"width_ratios": [4.8, 2.2]},
    )

    x_years = TARGET_YEARS
    for service_name in trend.columns:
        ax.plot(
            x_years,
            trend[service_name].tolist(),
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=service_name,
        )

    ax.set_title("Top 10 Services by Annual Count (2022-2025)", fontsize=20)
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel("Count", fontsize=15)
    ax.set_xticks(x_years)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.axis("off")
    ax_legend.legend(
        handles,
        labels,
        title="Service",
        fontsize=12,
        title_fontsize=14,
        loc="center left",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_IMAGE, dpi=220)
    plt.close()


def main() -> None:
    trend = build_top10_count_trend()
    plot_trend(trend)
    print(f"Generated: {OUTPUT_IMAGE.name}")


if __name__ == "__main__":
    main()
