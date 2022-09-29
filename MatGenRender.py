from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd

TestData = "MatGenTest.csv"
pdtest = pd.read_csv(TestData)
print(pdtest)

