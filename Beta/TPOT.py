import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("../TrainData.csv", index_col=0)

# Separate the features and target
X = data.drop(columns=["Metal Symbol", "X Symbol", "Energy"])
y = data["Energy"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create an instance of TPOTRegressor and specify parameters
tpot = TPOTRegressor(generations=20, population_size=500, verbosity=2, random_state=42, n_jobs=-1)

# Fit the model
tpot.fit(X_train, y_train)

# Evaluate the model
print("Score:", tpot.score(X_test, y_test))

# Export the best pipeline as a Python script
tpot.export("tpot_best_pipeline.py")
