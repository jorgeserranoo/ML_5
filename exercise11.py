import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

def load_and_preprocess_data(file_path, scaler_type='standard'):
    """
    Load the data and apply the specified scaling method
    """
    data = pd.read_csv(file_path)
    
    # Choose scaler based on parameter
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:  # minmax
        scaler = MinMaxScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(data)
    return scaled_data, data.columns

def analyze_clusters(data, max_clusters=16):
    """
    Perform clustering analysis with different numbers of clusters
    Returns inertia and silhouette scores
    """
    # Use powers of 2 for number of clusters
    n_clusters_list = [2**i for i in range(1, int(np.log2(max_clusters))+1)]
    inertias = []
    silhouette_scores = []
    
    for n_clusters in n_clusters_list:
        # Initialize and fit KMeans with k-means++ initialization for better starting centroids
        kmeans = KMeans(n_clusters=n_clusters, 
                       init='k-means++',
                       n_init=10,
                       random_state=42)
        
        # Fit the model and store inertia
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        
        print(f"Clusters: {n_clusters}, Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {silhouette_avg:.3f}")
    
    return n_clusters_list, inertias, silhouette_scores

def plot_analysis(n_clusters_list, inertias, silhouette_scores):
    """
    Create plots for both the elbow method and silhouette analysis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow method plot
    ax1.plot(n_clusters_list, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    # Use 'log' scale instead of 'log2'
    ax1.set_xscale('log')
    
    # Silhouette score plot
    ax2.plot(n_clusters_list, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    # Use 'log' scale instead of 'log2'
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    state_data, columns = load_and_preprocess_data('states_data.csv', scaler_type='standard')
    
    # Perform clustering analysis
    n_clusters_list, inertias, silhouette_scores = analyze_clusters(state_data)
    
    # Plot results
    plot_analysis(n_clusters_list, inertias, silhouette_scores)
    
    # Find best number of clusters based on silhouette score
    best_n_clusters = n_clusters_list[np.argmax(silhouette_scores)]
    print(f"\nBest number of clusters based on silhouette score: {best_n_clusters}")
    
    # Fit final model with best number of clusters
    final_kmeans = KMeans(n_clusters=best_n_clusters, 
                         init='k-means++',
                         n_init=10,
                         random_state=42)
    final_kmeans.fit(state_data)