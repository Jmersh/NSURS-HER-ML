# Imports
from MatGenRender import pdarray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from yellowbrick.features import PCA as ybPCA

warnings.filterwarnings('ignore')

# PCA 20
X = pdarray
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

pca_20 = PCA(n_components=20, random_state=2020)
pca_20.fit(X_scaled)
X_pca_20 = pca_20.transform(X_scaled)

# Should be 100% of variance explained
print("Variance by all 20 components =",
      sum(pca_20.explained_variance_ratio_ * 100))

# Plot showing number of components vs explained variance

plt.plot(np.cumsum(pca_20.explained_variance_ratio_))
plt.title("Number of components vs explained variance", pad=15)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()

print("Variance by all 2 components =",
      np.cumsum(pca_20.explained_variance_ratio_ * 100)[2])
print("Variance by all 3 components =",
      np.cumsum(pca_20.explained_variance_ratio_ * 100)[3])

# PCA 2

pca_2 = PCA(n_components=2, random_state=2020)
pca_2.fit(X_scaled)
X_pca_2 = pca_2.transform(X_scaled)

# Kmeans

kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(X_pca_2)

# 2D, 2 component scatter plot
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], s=70, hue=kmeans2.labels_, palette=['green', 'blue', 'red'])
plt.title("2D Scatter-plot: 78.15% variability captured", pad=15)
plt.xlabel("First component")
plt.ylabel("Second component")
plt.show()

# PCA 3
pca_3 = PCA(n_components=3, random_state=2020)
pca_3.fit(X_scaled)
X_pca_3 = pca_3.transform(X_scaled)

kmeans3 = KMeans(n_clusters=20)
kmeans3.fit(X_pca_3)

# Yellowbrick PCA visualizer

visualizer_3 = ybPCA(scale=True, projection=3,
                     classes=['g1', 'g2', 'g3'],
                     random_state=2020,
                     colors=['red', 'blue', 'green'])
visualizer_3.fit_transform(X, kmeans3.labels_)
visualizer_3.show()
