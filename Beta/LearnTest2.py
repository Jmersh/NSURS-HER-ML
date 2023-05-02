import os
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

# Load training data and data to be predicted from csv files
TrainData = pd.read_csv('../TrainData.csv', index_col=[0])
MatGenData = pd.read_csv('../MatGenOutput.csv', index_col=[0])

# Select the columns to be used as X and Y variables
X = TrainData.iloc[:, list(range(1, 22, 2))]
Y = TrainData.values[:, 22]
XMat = MatGenData.iloc[:, list(range(1, 22, 2))]
m = len(Y)

# Include material names in the data
material_names_train = TrainData.index.to_numpy().reshape(-1, 1)
material_names_matgen = MatGenData.index.to_numpy().reshape(-1, 1)

print(f'Total no of training examples (m) = {m}\n')


class GenModel:
    def __init__(self, model, n):
        self.model = model
        self.n = n
        self.X, self.y = make_classification(n_samples=1000, n_features=30, n_informative=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.p = self.model.fit(self.X_train, self.y_train).predict(self.X_test)
        self.TotalTraining = pd.DataFrame(self.X_test)
        self.TotalTraining["Actual E0"] = self.y_test
        self.TotalTraining["Predicted E0"] = self.p
        training_cols = ["Feature " + str(i+1) for i in range(30)]
        training_cols.append("Actual E0")
        training_cols.append("Predicted E0")
        self.TotalTraining.columns = training_cols
        self.save_arrays(model, 1)




    def generate_models(self, model, n, p):
        for ModelN in range(n):
            X_train, X_test, Y_train, Y_test, material_names_train_split, material_names_test_split = train_test_split(
                X, Y, material_names_train, random_state=ModelN
            )

            # Train the model with the training dataset
            model.fit(X_train, Y_train)

            # Evaluate the model on the training dataset and calculate the RMSE
            TrGMprede = model.predict(X_train).reshape(-1, 1)
            TrGMacte = Y_train.reshape(-1, 1)
            TrGMrmse = np.sqrt(mean_squared_error(TrGMprede, TrGMacte))

            # Evaluate the model on the testing dataset and calculate the RMSE
            TeGMprede = model.predict(X_test).reshape(-1, 1)
            TeGMacte = Y_test.reshape(-1, 1)
            TeGMrmse = np.sqrt(mean_squared_error(TeGMprede, TeGMacte))

            # Add RMSE values to the Totalrmse dataframe
            self.Totalrmse.loc[len(self.Totalrmse.index)] = [TrGMrmse, TeGMrmse]

            # Concatenate results for each iteration
            self.TotalTraining = pd.concat([self.TotalTraining, pd.DataFrame(np.hstack([material_names_train_split, TrGMacte, TrGMprede]))], axis=1)
            self.TotalTesting = pd.concat([self.TotalTesting, self.TotalTesting, pd.DataFrame(np.hstack([material_names_test_split, TeGMacte, TeGMprede]))], axis=1)

            if p == 1:
                PredMat = model.predict(XMat).reshape(-1, 1)
                self.TotalPred = pd.concat([self.TotalPred, pd.DataFrame(np.hstack([material_names_matgen, PredMat]))], axis=1)

    def save_arrays(self, model, p):

        training_cols = ["Material Name"] + [f"Tr Actual E{i}" for i in range(len(X.columns))] + [f"Tr Predicted E{i}"
                                                                                                  for i in
                                                                                                  range(len(X.columns))]
        testing_cols = ["Material Name"] + [f"Te Actual E{i}" for i in range(len(X.columns))] + [f"Te Predicted E{i}"
                                                                                                 for i in
                                                                                                 range(len(X.columns))]
        pred_cols = ["Material Name"] + [f"Predicted E{i}" for i in range(len(XMat.columns))]

        self.TotalTraining.columns = training_cols
        self.TotalTesting.columns = testing_cols
        self.TotalPred.columns = pred_cols

        self.TotalTraining.to_csv(f'{model.__class__.__name__}_Training.csv', index=False)
        self.TotalTesting.to_csv(f'{model.__class__.__name__}_Testing.csv', index=False)
        self.Totalrmse.to_csv(f'{model.__class__.__name__}_RMSE.csv', index=False)

        if p == 1:
            self.TotalPred.to_csv(f'{model.__class__.__name__}_Pred.csv', index=False)


class Timer:
    def __init__(self):
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Time taken: {time.time() - self.start_time:.2f} seconds\n")

if __name__ == "__main__":
    # Initialize and train different models
    models = [
        LinearRegression(),
        RidgeCV(alphas=[0.1, 1.0, 10.0]),
        make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
        GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel()),
        KernelRidge(alpha=1.0),
        DecisionTreeRegressor(max_depth=2),
        RandomForestRegressor(max_depth=2, n_estimators=10),
        MLPRegressor(hidden_layer_sizes=(100,), alpha=1e-5, random_state=1),
    ]

    for model in models:
        with Timer():
            print(f"Training and evaluating {model.__class__.__name__}")
            gm = GenModel(model, n=10)

