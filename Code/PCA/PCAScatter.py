import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Service columns to keep (common services across all files)
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

# Load and combine data from all years
def load_service_data():
    data_files = ['../22.csv', '../23.csv', '../24.csv', '../25.csv']
    dfs = []
    
    for file in data_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# Process the data for PCA
def process_data(df):
    # Get the Age column
    age_col = df['Age'].copy()
    
    # Select only service columns that exist in the dataframe
    available_cols = [col for col in SERVICE_COLUMNS if col in df.columns]
    service_cols = df[available_cols].copy()
    
    # Convert to binary (1 if service is mentioned, 0 if empty/NaN)
    service_data = service_cols.notna() & (service_cols != '')
    service_data = service_data.astype(int)
    
    # Remove rows with no services (all zeros)
    service_data = service_data[service_data.sum(axis=1) > 0].reset_index(drop=True)
    age_col = age_col[service_data.index].reset_index(drop=True)
    
    return service_data, age_col

# Define age group colors
def get_age_colors():
    age_order = [
        '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75 or older'
    ]
    colors = {
        '18-24': '#1f77b4',      # blue
        '25-34': '#2ca02c',      # green
        '35-44': '#ff7f0e',      # orange
        '45-54': '#d62728',      # red
        '55-64': '#9467bd',      # purple
        '65-74': '#8c564b',      # brown
        '75 or older': '#e377c2' # pink
    }
    return colors, age_order

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading service usage data...")
    df = load_service_data()
    
    # Process data
    print("Processing data for PCA...")
    X, age = process_data(df)
    
    print(f"Data shape: {X.shape}")
    print(f"Service columns: {len(X.columns)}")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"PC2 variance explained: {pca.explained_variance_ratio_[1]:.1%}")

    # Save PCA loadings for the first two principal components
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=['PC1', 'PC2']
    )
    loadings_df.to_csv('PCA_Loadings_PC1_PC2.csv', index_label='Service')

    print("\nTop 5 absolute loadings for PC1:")
    print(loadings_df['PC1'].abs().sort_values(ascending=False).head(5))
    print("\nTop 5 absolute loadings for PC2:")
    print(loadings_df['PC2'].abs().sort_values(ascending=False).head(5))
    print("\nSaved: PCA_Loadings_PC1_PC2.csv")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Get color mapping
    age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75 or older', 'Prefer not to answer']
    colors_map = {
        '18-24': '#1f77b4',      # blue
        '25-34': '#2ca02c',      # green
        '35-44': '#ff7f0e',      # orange
        '45-54': '#d62728',      # red
        '55-64': '#9467bd',      # purple
        '65-74': '#8c564b',      # brown
        '75 or older': '#e377c2', # pink
        'Prefer not to answer': '#7f7f7f'  # gray
    }
    
    # Plot each age group
    for age_group in age_order:
        mask = age == age_group
        if mask.any():
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                      label=age_group, 
                      color=colors_map.get(age_group, '#999999'),
                      alpha=0.6, 
                      s=50,
                      edgecolors='none')
    
    # Add reference lines through origin
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA of Service Usage by Age Group')
    
    # Legend
    ax.legend(title='Age Group', loc='best', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig('PCA_Scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
