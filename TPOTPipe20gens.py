import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('TrainData.csv', sep=',', index_col=0)
tpot_data = tpot_data.drop(tpot_data.columns[[0, 2]], axis=1)
features = tpot_data.drop('Energy', axis=1)


training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Energy'], random_state=42)

# Average CV score on the training set was: -0.06394220063127867, tpot = TPOTRegressor(generations=20, population_size=500, verbosity=2, random_state=42, n_jobs=-1)
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=4, min_child_weight=11, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.25, verbosity=0)),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="huber", max_depth=10, max_features=0.4, min_samples_leaf=20, min_samples_split=17, n_estimators=100, subsample=0.9500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
