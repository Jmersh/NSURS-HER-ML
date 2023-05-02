# Importing our necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Creating an array using numpy of 10 rows and 2 columns.
X = np.array([[3,3],
     [8,18],
     [11,12],
     [43,10],
     [25,45],
     [90,70],
     [67,80],
     [55,78],
     [55,52],
     [70,99],])
# Plotting our array, this plots the first column against the second column.
plt.scatter(X[:, 0], X[:, 1], label='True Position')

# Calling pyplot to visually display our plot.
plt.show()

# Defining how many clusters we would like to use and begin clustering.
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Printing our centroid values for our two clusters.
print(kmeans.cluster_centers_)

# Printing which cluster or label each data point was assigned.
print(kmeans.labels_)

# Plotting our array again but visualizing which cluster they have been sorted into.
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')

# Plotting the centroid of our two clusters in black.
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')

# Calling pyplot to visually display our plot.
plt.show()