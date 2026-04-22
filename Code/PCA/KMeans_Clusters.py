# KMeans_Clusters.py
# K-means clustering on PCA-transformed service usage data
# Visualize clusters in the PCA 2D space
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

# Service columns to keep
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
    
    # Convert to binary
    service_data = service_cols.notna() & (service_cols != '')
    service_data = service_data.astype(int)
    
    # Remove rows with no services
    service_data = service_data[service_data.sum(axis=1) > 0].reset_index(drop=True)
    age_col = age_col[service_data.index].reset_index(drop=True)
    
    return service_data, age_col

if __name__ == "__main__":
    print("Loading and processing service usage data...")
    df = load_service_data()
    X, age = process_data(df)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Data shape: {X.shape}")
    print(f"PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
    
    # Find optimal number of clusters using elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    from sklearn.metrics import silhouette_score
    
    print("\nTesting cluster numbers 2-10...")
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_pca)
        inertias.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(X_pca, kmeans_temp.labels_))
        print(f"  k={k}: Inertia={kmeans_temp.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
    
    # Use k=4 as a good balance (can be adjusted based on elbow plot)
    optimal_k = 4
    print(f"\nUsing k={optimal_k} clusters")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Color map for clusters
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # Plot clusters
    for cluster_id in range(optimal_k):
        mask = clusters == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  label=f'Cluster {cluster_id+1} (n={mask.sum()})',
                  color=colors[cluster_id % len(colors)],
                  alpha=0.6,
                  s=50,
                  edgecolors='black',
                  linewidth=0.5)
    
    # Plot cluster centers
    centers_pca = kmeans.cluster_centers_
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
              marker='o',
              s=400,
              c='white',
              edgecolors='black',
              linewidth=2,
              label='Cluster Centers',
              zorder=5)
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'K-Means Clustering (k={optimal_k}) of Service Usage Patterns', fontsize=14)
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('KMeans_Clusters.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: KMeans_Clusters.png")
    
    # Save cluster assignments for further analysis
    cluster_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters,
        'Age': age
    })
    cluster_df.to_csv('cluster_assignments.csv', index=False)
    
    plt.close()
