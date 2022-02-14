# Importing our necessary libraries
import json

#import JSONStorage as JSONStorage
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def loadjson(path):
    with open(path) as f:
        data = json.load(f)
    # print(data)

    return np.asarray(data)


#loadjson('JSONStorage/Ti2B.json')

# Creates an array from the data loaded from our JSON file
test1 = np.array(loadjson('JSONStorage/Ti2B.json'))

plt.scatter(test1)
plt.show()