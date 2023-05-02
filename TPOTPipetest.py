import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
from sklearn.metrics import mean_squared_error, r2_score
import os
import time


def load_predict_data(file_path):
    predict_data = pd.read_csv(file_path, sep=',', index_col=0)
    predict_data = predict_data.drop(predict_data.columns[[0, 2]], axis=1)
    return predict_data

def save_predictions_to_csv(predictions, output_file):
    result_df = pd.DataFrame(predictions, columns=["Predicted_Energy"])
    result_df.to_csv(output_file, index=False)

# Load the training data
tpot_data = pd.read_csv('TrainData.csv', sep=',', index_col=0)
tpot_data = tpot_data.drop(tpot_data.columns[[0, 2]], axis=1)
features = tpot_data.drop('Energy', axis=1)

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Energy'], random_state=42)

exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=4, min_child_weight=11, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.25, verbosity=0)),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="huber", max_depth=10, max_features=0.4, min_samples_leaf=20, min_samples_split=17, n_estimators=100, subsample=0.9500000000000001)
)
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# Fit the model and measure the time taken
start_time = time.time()
exported_pipeline.fit(training_features, training_target)
fit_time = time.time() - start_time
print(f"Time taken to fit the model: {fit_time:.2f} seconds")

# Calculate and print RMSE for training data
training_predictions = exported_pipeline.predict(training_features)
training_rmse = np.sqrt(mean_squared_error(training_target, training_predictions))
print(f"Training RMSE: {training_rmse}")

# Calculate and print RMSE for testing data
testing_predictions = exported_pipeline.predict(testing_features)
testing_rmse = np.sqrt(mean_squared_error(testing_target, testing_predictions))
print(f"Testing RMSE: {testing_rmse}")

# Calculate and print R^2 scores for training and testing data
training_r2_score = r2_score(training_target, training_predictions)
testing_r2_score = r2_score(testing_target, testing_predictions)
print(f"Training R^2 Score: {training_r2_score}")
print(f"Testing R^2 Score: {testing_r2_score}")

# Load the data to predict energy values measure the time taken
predict_data = load_predict_data('MatGenOutput.csv')
start_time = time.time()
predictions = exported_pipeline.predict(predict_data)
pred_time = time.time() - start_time
print(f"Time taken to fit the model: {pred_time:.2f} seconds")

# Save the predicted energy values to a CSV file
save_predictions_to_csv(predictions, 'PredictedEnergyValues.csv')
print("Predicted energy values saved to 'PredictedEnergyValues.csv'")