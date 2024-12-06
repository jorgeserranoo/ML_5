import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
import seaborn as sns

def analyze_action_clusters():
    # Load and preprocess the action data
    # Load and preprocess the action data
    data = pd.read_csv('actions_data.csv')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Create linkage matrix using ward method
    linkage_matrix = scipy_linkage(scaled_data, method='ward')

    # Plot the dendrogram with adjusted parameters
    plt.figure(figsize=(15, 8))

    # Plot dendrogram with color threshold to show 5 clusters
    dendrogram_plot = dendrogram(
        linkage_matrix,
        truncate_mode='none',
        color_threshold=45,    # Adjusted to better show natural 5-cluster separation
        leaf_rotation=90,
        leaf_font_size=8
    )

    plt.title('Hierarchical Clustering Dendrogram for Action Space\n(Ward Linkage, 5 Clusters)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')

    # Add horizontal line at the natural separation point for 5 clusters
    plt.axhline(y=45, color='r', linestyle='--', label='Cluster Separation Level')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Perform the clustering with optimal parameters
    clustering = AgglomerativeClustering(
        n_clusters=5,  # Our optimal number of clusters
        linkage='ward'  # Our best performing linkage method
    )
    labels = clustering.fit_predict(scaled_data)
    
    # Create a DataFrame with the original data and cluster labels
    cluster_df = pd.DataFrame({
        'action': data.iloc[:, 0],
        'cluster': labels
    })
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    print("==================")
    for cluster in range(5):
        cluster_data = cluster_df[cluster_df['cluster'] == cluster]['action']
        print(f"\nCluster {cluster}:")
        print(f"Number of samples: {len(cluster_data)}")
        print(f"Mean action: {cluster_data.mean():.3f}")
        print(f"Std deviation: {cluster_data.std():.3f}")
        print(f"Min action: {cluster_data.min():.3f}")
        print(f"Max action: {cluster_data.max():.3f}")

if __name__ == "__main__":
    analyze_action_clusters()