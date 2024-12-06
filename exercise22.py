import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage as scipy_linkage
import random

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across all relevant libraries
    """
    random.seed(seed)
    np.random.seed(seed)

def load_and_preprocess_data(filepath, scaler_type='standard'):
    """
    Load and preprocess the action space data. This function remains the same
    but will now handle the action data which might have different characteristics.
    """
    # Load the action data
    data = pd.read_csv(filepath)
    
    # Select the scaler - for action data, both StandardScaler and MinMaxScaler could be appropriate
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Scale the data
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, data

def evaluate_clustering(data, n_clusters_range, linkage_methods=['ward', 'complete', 'average'], random_state=42):
    """
    Evaluate different clustering parameters using silhouette score.
    This is particularly important for action space as it might have different optimal parameters.
    """
    results = []
    
    for linkage in linkage_methods:
        for n_clusters in n_clusters_range:
            # Create and fit the clustering model with fixed random_state
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric='euclidean'  # Keeping euclidean distance for consistency
            )
            labels = model.fit_predict(data)
            
            # Calculate silhouette score
            sil_score = silhouette_score(data, labels)
            
            results.append({
                'linkage': linkage,
                'n_clusters': n_clusters,
                'silhouette_score': sil_score
            })
    
    return pd.DataFrame(results)

def plot_dendrogram(data, linkage_method='ward', title='Hierarchical Clustering Dendrogram for Action Space'):
    """
    Create and plot dendrogram specifically for action space data
    """
    plt.figure(figsize=(10, 7))
    
    # Create linkage matrix using renamed scipy_linkage
    linkage_matrix = scipy_linkage(data, method=linkage_method)
    
    # Plot dendrogram with more informative title for action space
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

# Main analysis
if __name__ == "__main__":
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Load and preprocess action data
    scaled_data, raw_data = load_and_preprocess_data('actions_data.csv', scaler_type='standard')
    
    # Define parameter ranges
    # For action space, we might want to try a different range of clusters
    # since the action space might have different characteristics
    n_clusters_range = range(2, 11)
    linkage_methods = ['ward', 'complete', 'average']
    
    # Evaluate different parameters
    results_df = evaluate_clustering(scaled_data, n_clusters_range, linkage_methods, random_state=42)
    
    # Find best parameters
    best_result = results_df.loc[results_df['silhouette_score'].idxmax()]
    print("\nBest parameters for action space clustering:")
    print(f"Linkage method: {best_result['linkage']}")
    print(f"Number of clusters: {best_result['n_clusters']}")
    print(f"Silhouette score: {best_result['silhouette_score']:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for linkage in linkage_methods:
        linkage_results = results_df[results_df['linkage'] == linkage]
        plt.plot(linkage_results['n_clusters'], 
                linkage_results['silhouette_score'], 
                marker='o', 
                label=linkage)
    
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters (Action Space)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot dendrogram for best linkage method
    plot_dendrogram(scaled_data, linkage_method=best_result['linkage'])

    # Additional visualization for action space (if it's 1D data)
    if scaled_data.shape[1] == 1:
        best_model = AgglomerativeClustering(
            n_clusters=best_result['n_clusters'],
            linkage=best_result['linkage']
        )
        labels = best_model.fit_predict(scaled_data)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(scaled_data, np.zeros_like(scaled_data), c=labels, cmap='viridis')
        plt.title(f'Action Space Clusters (n_clusters={best_result["n_clusters"]})')
        plt.xlabel('Action Value')
        plt.ylabel('Arbitrary Axis (for visualization)')
        plt.colorbar(label='Cluster')
        plt.show()