import pymatgen
from elements.elements import Fe
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd

TestData = "MatGenOutput.csv"  # Declare CSV file to be used in pandas dataframe
pdtest = pd.read_csv(TestData)  # draws dataframe from CSV
print(pdtest)  # prints dataframe from pandas

print(pymatgen.core.periodic_table.ElementBase(Fe).electron_affinity)
