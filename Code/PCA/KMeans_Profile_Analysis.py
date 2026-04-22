# KMeans_Profile_Analysis.py
# Analyze and compare cluster profiles - what services define each cluster
# Simplified version with clean bar charts
# 4/13/2026

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Service columns
SERVICE_COLUMNS = [
    'Acupuncture',
    'Adult Counseling - Individual',
    'Adult Counseling - Support Groups',
    'Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, etc.)',
    'Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)',
    'Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)',
    'Dempsey Soup Program',
    'Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)',
    'Expressive Arts Workshop (e.g., Writing Through Cancer, Music for Wellness, etc.)',
    'Individual Counseling',
    'Legacy & Life Reflections',
    'Massage Therapy',
    'Movement & Fitness Consult',
    'Movement & Fitness Classes/Workshops',
    'Nutrition Consult',
    'Nutrition Classes/Workshops',
    'Reiki Session',
    'Support Groups',
    'Wig & Headwear Consult',
    'Youth & Family Counseling',
    'Youth & Family Classes/Workshops'
]

DISPLAY_NAMES = {
    'Acupuncture': 'Acupuncture',
    'Adult Counseling - Individual': 'Individual Counseling',
    'Adult Counseling - Support Groups': 'Support Groups',
    'Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, etc.)': 'Comfort Items',
    'Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)': 'Complementary Workshops',
    'Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)': 'Dempsey Dogs',
    'Dempsey Soup Program': 'Dempsey Soup Program',
    'Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)': 'Educational Workshop',
    'Expressive Arts Workshop (e.g., Writing Through Cancer, Music for Wellness, etc.)': 'Expressive Arts Workshop',
    'Individual Counseling': 'Individual Counseling',
    'Legacy & Life Reflections': 'Legacy & Life Reflections',
    'Massage Therapy': 'Massage Therapy',
    'Movement & Fitness Consult': 'Movement & Fitness Consult',
    'Movement & Fitness Classes/Workshops': 'Movement & Fitness Classes',
    'Nutrition Consult': 'Nutrition Consult',
    'Nutrition Classes/Workshops': 'Nutrition Classes',
    'Reiki Session': 'Reiki Session',
    'Support Groups': 'Support Groups',
    'Wig & Headwear Consult': 'Wig & Headwear Consult',
    'Youth & Family Counseling': 'Youth & Family Counseling',
    'Youth & Family Classes/Workshops': 'Youth & Family Classes'
}

def load_service_data():
    data_files = [
        os.path.join(SCRIPT_DIR, '../22.csv'),
        os.path.join(SCRIPT_DIR, '../23.csv'),
        os.path.join(SCRIPT_DIR, '../24.csv'),
        os.path.join(SCRIPT_DIR, '../25.csv'),
    ]
    dfs = []
    
    for file in data_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def process_data(df):
    age_col = df['Age'].copy()
    available_cols = [col for col in SERVICE_COLUMNS if col in df.columns]
    service_cols = df[available_cols].copy()
    
    service_data = service_cols.notna() & (service_cols != '')
    service_data = service_data.astype(int)
    
    service_data = service_data[service_data.sum(axis=1) > 0].reset_index(drop=True)
    age_col = age_col[service_data.index].reset_index(drop=True)
    
    return service_data, age_col, available_cols

if __name__ == "__main__":
    print("Loading and processing service usage data...")
    df = load_service_data()
    X, age, available_cols = process_data(df)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply K-means with k=4
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    # Calculate cluster profiles (mean service usage per cluster)
    X['Cluster'] = clusters
    cluster_profiles = X.groupby('Cluster')[available_cols].mean()
    
    # Create simple bar chart with top services per cluster
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors_clusters = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for cluster_id in range(optimal_k):
        ax = axes[cluster_id]
        
        # Get top 8 services for this cluster
        top_services = cluster_profiles.iloc[cluster_id].nlargest(8)
        
        # Shorten service names for display
        short_names = [DISPLAY_NAMES.get(name, name) for name in top_services.index]
        
        # Create bar chart
        bars = ax.barh(range(len(top_services)), top_services.values * 100, 
                       color=colors_clusters[cluster_id], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Formatting
        ax.set_yticks(range(len(top_services)))
        ax.set_yticklabels(short_names, fontsize=10)
        ax.set_xlabel('Usage Rate (%)', fontsize=11)
        ax.set_xlim(0, 100)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_services.values)):
            ax.text(val * 100 + 2, i, f'{val*100:.0f}%', va='center', fontsize=9)
        
        # Title with cluster size
        cluster_size = sum(clusters == cluster_id)
        ax.set_title(f'Cluster {cluster_id + 1} (n={cluster_size})', 
                    fontsize=12, pad=10)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Top Services by Cluster', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig('KMeansProfiles.png', dpi=300, bbox_inches='tight')
    
    plt.close()
