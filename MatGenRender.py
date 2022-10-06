import pymatgen
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd

# Declare CSV file to be used in pandas dataframe
TestData = "MatGenOutput.csv"

# draws dataframe from CSV
pdarray = pd.read_csv(TestData, index_col=[0],  usecols=[0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 , 14, 15, 16, 17, 18, 19, 20, 21, 22])

# prints dataframe from pandas
print(pdarray)
print(pdarray.shape)
