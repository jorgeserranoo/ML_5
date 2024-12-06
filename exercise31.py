import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load our state space data which contains x, y positions and angular velocity
state_data = pd.read_csv('states_data.csv')

# Extract our three features into a numpy array for processing
X = state_data[['x', 'y', 'Angular_velocity']].values

# Standardize the features to have zero mean and unit variance
# This is crucial for PCA since our features are on different scales:
# - x, y are positions
# - Angular_velocity is in different units entirely
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 3 components as specified in the exercise
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Get the explained variance ratios for each component
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Create bar plot of individual explained variance ratios
plt.figure(figsize=(10, 6))
bars = plt.bar(range(1, 4), explained_variance_ratio, color='skyblue')
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('Explained Variance Ratio by Principal Component', fontsize=14)
plt.xticks([1, 2, 3])
plt.grid(True, alpha=0.3)

# Add percentage labels on top of each bar for better readability
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height*100:.1f}%',
             ha='center', va='bottom')
plt.show()

# Create line plot of cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, 4), cumulative_variance_ratio, marker='o', color='blue', linewidth=2)
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
plt.title('Cumulative Explained Variance Ratio vs Number of Components', fontsize=14)
plt.xticks([1, 2, 3])
plt.grid(True, alpha=0.3)

# Add percentage labels for each point in the line plot
for i, ratio in enumerate(cumulative_variance_ratio):
    plt.text(i+1, ratio, f'{ratio*100:.1f}%', 
             ha='center', va='bottom')
plt.show()

# Print the detailed variance analysis
print("\nDetailed PCA Analysis:")
print("-" * 50)
print("\nExplained variance ratio for each component:")
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"PC{i}: {ratio:.4f} ({ratio*100:.2f}%)")

print("\nCumulative explained variance ratio:")
for i, ratio in enumerate(cumulative_variance_ratio, 1):
    print(f"First {i} components: {ratio:.4f} ({ratio*100:.2f}%)")