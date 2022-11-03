from MatGenRender import pdarray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from yellowbrick.features import PCA as ybPCA


plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca_2[:, 0], y=X_pca_2[:, 1], s=70, hue=kmeans2.labels_, palette=['green', 'blue', 'red'])
plt.title("2D Scatter-plot: 78.15% variability captured", pad=15)
plt.xlabel("First component")
plt.ylabel("Second component")
plt.show()