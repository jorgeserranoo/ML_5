import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def visualize_state_clusters(data, n_clusters=8):
    """
    Create 3D visualization of state space clusters
    """
    # Prepare the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with different colors for each cluster
    scatter = ax.scatter(scaled_data[:, 0], 
                        scaled_data[:, 1], 
                        scaled_data[:, 2],
                        c=cluster_labels, 
                        cmap='tab10',
                        alpha=0.6)
    
    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Angular Velocity')
    ax.set_title('State Space Clusters')
    
    # Add colorbar
    plt.colorbar(scatter, label='Cluster')
    plt.show()

def visualize_action_clusters(data, n_clusters=4):
    """
    Create visualization of action space clusters
    """
    # Prepare the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Create histogram of actions by cluster
    plt.figure(figsize=(12, 6))
    
    # Plot distribution of actions for each cluster
    for i in range(n_clusters):
        cluster_actions = scaled_data[cluster_labels == i]
        plt.hist(cluster_actions, bins=30, alpha=0.5, label=f'Cluster {i}')
    
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')
    plt.title('Action Space Clusters Distribution')
    plt.legend()
    plt.show()
    
    # Print cluster statistics
    for i in range(n_clusters):
        cluster_actions = scaled_data[cluster_labels == i]
        print(f"\nCluster {i} statistics:")
        print(f"Mean: {np.mean(cluster_actions):.3f}")
        print(f"Std: {np.std(cluster_actions):.3f}")
        print(f"Size: {len(cluster_actions)}")



# Main execution
if __name__ == "__main__":
    # Load and analyze state space
    state_data = pd.read_csv('states_data.csv')
    print("Visualizing State Space Clusters (k=8):")
    visualize_state_clusters(state_data)
    
    # Load and analyze action space
    action_data = pd.read_csv('actions_data.csv')
    print("\nVisualizing Action Space Clusters (k=4):")
    visualize_action_clusters(action_data)