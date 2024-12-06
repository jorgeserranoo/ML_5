import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar y preparar los datos
state_data = pd.read_csv('states_data.csv')
X = state_data[['x', 'y', 'Angular_velocity']].values

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear una figura con dos subplots: 3D y 2D
plt.figure(figsize=(15, 6))

# 1. Visualización 3D de los datos originales
ax1 = plt.subplot(121, projection='3d')
scatter1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
                      c=X_scaled[:, 2], cmap='viridis', alpha=0.6)
ax1.set_xlabel('X Position (scaled)')
ax1.set_ylabel('Y Position (scaled)')
ax1.set_zlabel('Angular Velocity (scaled)')
ax1.set_title('Original Data in 3D')
plt.colorbar(scatter1, ax=ax1, label='Angular Velocity')

# 2. Aplicar PCA con 2 componentes y visualizar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualización 2D de los datos reducidos
ax2 = plt.subplot(122)
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=X_scaled[:, 2], cmap='viridis', alpha=0.6)
ax2.set_xlabel('First Principal Component')
ax2.set_ylabel('Second Principal Component')
ax2.set_title('PCA-Reduced Data in 2D')
plt.colorbar(scatter2, ax=ax2, label='Angular Velocity')

plt.tight_layout()
plt.show()

# Análisis de la varianza explicada
print("\nAnálisis de la reducción de dimensionalidad:")
print("-" * 50)
print("\nVarianza explicada por cada componente:")
for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
    print(f"Componente Principal {i}: {ratio:.4f} ({ratio*100:.2f}%)")

print(f"\nVarianza total explicada con 2 componentes: {sum(pca.explained_variance_ratio_):.4f} "
      f"({sum(pca.explained_variance_ratio_)*100:.2f}%)")

# Análisis de las componentes principales
print("\nContribución de las variables originales a las componentes principales:")
print("-" * 50)
variables = ['X Position', 'Y Position', 'Angular Velocity']
for i, component in enumerate(pca.components_, 1):
    print(f"\nComponente Principal {i}:")
    for var, weight in zip(variables, component):
        print(f"{var}: {weight:.4f}")