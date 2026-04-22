# PCALoading.py - Perform PCA on service usage data and visualize the loadings for the top contributing services to each principal component.

import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SERVICE_COLUMNS = [
    "Acupuncture",
    "Adult Counseling - Individual",
    "Adult Counseling - Support Groups",
    "Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, etc.)",
    "Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)",
    "Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)",
    "Dempsey Soup Program",
    "Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)",
    "Expressive Arts Workshop (e.g., Writing Through Cancer, Music for Wellness, etc.)",
    "Individual Counseling",
    "Legacy & Life Reflections",
    "Massage Therapy",
    "Movement & Fitness Consult",
    "Movement & Fitness Classes/Workshops",
    "Nutrition Consult",
    "Nutrition Classes/Workshops",
    "Reiki Session",
    "Support Groups",
    "Wig & Headwear Consult",
    "Youth & Family Counseling",
    "Youth & Family Classes/Workshops",
]

DISPLAY_NAMES = {
    "Acupuncture": "Acupuncture",
    "Adult Counseling - Individual": "Individual Counseling",
    "Adult Counseling - Support Groups": "Support Groups",
    "Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, etc.)": "Comfort Items",
    "Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)": "Complementary Workshops",
    "Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)": "Dempsey Dogs",
    "Dempsey Soup Program": "Dempsey Soup Program",
    "Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)": "Educational Workshop",
    "Expressive Arts Workshop (e.g., Writing Through Cancer, Music for Wellness, etc.)": "Expressive Arts Workshop",
    "Individual Counseling": "Individual Counseling",
    "Legacy & Life Reflections": "Legacy & Life Reflections",
    "Massage Therapy": "Massage Therapy",
    "Movement & Fitness Consult": "Movement & Fitness Consult",
    "Movement & Fitness Classes/Workshops": "Movement & Fitness Classes",
    "Nutrition Consult": "Nutrition Consult",
    "Nutrition Classes/Workshops": "Nutrition Classes",
    "Reiki Session": "Reiki Session",
    "Support Groups": "Support Groups",
    "Wig & Headwear Consult": "Wig & Headwear Consult",
    "Youth & Family Counseling": "Youth & Family Counseling",
    "Youth & Family Classes/Workshops": "Youth & Family Classes",
}


def load_data():
    files = [
        os.path.join(SCRIPT_DIR, "../22.csv"),
        os.path.join(SCRIPT_DIR, "../23.csv"),
        os.path.join(SCRIPT_DIR, "../24.csv"),
        os.path.join(SCRIPT_DIR, "../25.csv"),
    ]

    frames = []
    for file_path in files:
        if os.path.exists(file_path):
            frames.append(pd.read_csv(file_path))

    if not frames:
        raise ValueError("No CSV files found.")

    return pd.concat(frames, ignore_index=True)


def prepare_matrix(df):
    available = [c for c in SERVICE_COLUMNS if c in df.columns]
    X = df[available].notna() & (df[available] != "")
    X = X.astype(int)
    X = X[X.sum(axis=1) > 0].reset_index(drop=True)
    return X


def main():
    df = load_data()
    X = prepare_matrix(df)

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=["PC1", "PC2"],
    )

    loadings["Service"] = [DISPLAY_NAMES.get(s, s) for s in loadings.index]
    loadings.to_csv(os.path.join(SCRIPT_DIR, "PCA_Loadings.csv"), index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    for ax, col in zip(axes, ["PC1", "PC2"]):
        temp = loadings[["Service", col]].copy()
        top_index = temp[col].abs().nlargest(10).sort_values().index
        temp = temp.loc[top_index]

        ax.barh(temp["Service"], temp[col], color="#6aaed6")
        ax.set_title(f"{col} loadings")
        ax.set_xlabel("Loading")
        ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "PCALoading.png"), dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
