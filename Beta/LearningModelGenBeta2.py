import os

import numpy as np
import pandas as pd
import sklearn
from IPython.core.display_functions import display
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from inspect import signature
from sklearn.pipeline import make_pipeline
import time

# Load csv files of training data and data you'd like to predict.

TrainData = pd.read_csv('outputClean.csv', index_col=[0])
MatGenData = pd.read_csv('MatGenOutputLatticeClean.csv', index_col=[0])

# Choose which columns you'd like to use as your X and Y variables.

X = TrainData.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]]
Y = TrainData.values[:, 22]

XMat = MatGenData.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]

# X = TrainData.values[:, [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
# Y = TrainData.values[:, 23]
m = len(Y)

# Small debug print to show how many training examples.

print('Total no of training examples (m) = %s \n' % (m))


# Function portion of code. These are built to automate features.

class GenModel:
    # Establish empty data frames that get concated to after every for loop.
    TotalTraining = pd.DataFrame(columns=['Tr Predicted E', 'Tr Actual E'])
    TotalTesting = pd.DataFrame(columns=['Te Predicted E', 'Te Actual E'])
    Totalrmse = pd.DataFrame(columns=['Training RMSE', 'Testing RMSE'])
    TotalPred = pd.DataFrame(columns=['Predicted E'])

    # TotalIndice = pd.DataFrame(columns=['Name'])

    def __init__(self, model, n=10, p=0):
        for ModelN in range(n):
            # Splitting training and testing data, random_state value for replicability.
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=ModelN)
            # Indice = pd.DataFrame([X_train.index.values, X_test.index.values, Y_train.index.values, Y_test.index.values], columns=['X_train Ind' + str(ModelN), 'X_test Ind' + str(ModelN), 'Y_train Ind' + str(ModelN), 'Y_test Ind' + str(ModelN)])
            # self.TotalIndice = pd.concat([self.TotalIndice, Indice], axis=1)

            # This is a currently unused feature to scale data.
            scaler = StandardScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.fit_transform(X_test)

            # Fits your input model variable to training data.
            model.fit(X_train, Y_train)

            # Training Array Maker
            TrGMprede = pd.DataFrame(model.predict(X_train), columns=['Tr Predicted E' + str(ModelN)])
            TrGMacte = pd.DataFrame(Y_train, columns=['Tr Actual E' + str(ModelN)])
            # TrGMacte = TrGMacte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new
            # dataframe
            TrGM_actual_vs_predicted = pd.concat([TrGMacte, TrGMprede], axis=1)
            self.TotalTraining = pd.concat([self.TotalTraining, TrGM_actual_vs_predicted], axis=1)
            TrGMrmse = np.sqrt(mean_squared_error(TrGMprede, TrGMacte))

            # Test Array Maker
            TeGMprede = pd.DataFrame(model.predict(X_test), columns=['Te Predicted E' + str(ModelN)])
            TeGMacte = pd.DataFrame(Y_test, columns=['Te Actual E' + str(ModelN)])
            # TeGMacte = TeGMacte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new
            # dataframe
            TeGM_actual_vs_predicted = pd.concat([TeGMacte, TeGMprede], axis=1)
            self.TotalTesting = pd.concat([self.TotalTesting, TeGM_actual_vs_predicted], axis=1)
            TeGMrmse = np.sqrt(mean_squared_error(TeGMprede, TeGMacte))

            self.Totalrmse.loc[len(self.Totalrmse.index)] = [TrGMrmse, TeGMrmse]

            # If the p variable is equal to 1 then it will also use the model to predict the data and concat to data
            # frame.
            if p == 1:
                model.predict(XMat)
                PredMat = pd.DataFrame(model.predict(XMat), columns=['Predicted E'])
                self.TotalPred = pd.concat([self.TotalPred, PredMat], axis=1)
            else:
                pass
            continue

    # This function returns the arrays from the init function. Should maybe change function structure in future.
    def ReturnArray(self, model, p=0):
        self.Totalrmse.loc['mean'] = self.Totalrmse.mean()
        self.TotalTesting.dropna(axis='columns', inplace=True)
        self.TotalTraining.dropna(axis='columns', inplace=True)

        # This prints arrays for debuging
        # print(self.TotalTraining)
        # print(self.TotalTesting)
        # print(self.Totalrmse)

        # print model completion
        print(model, 'generation complete.')
        # saving arrays for future use
        isExist = os.path.exists('Models/' + model)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs('Models/' + model)
        self.TotalTraining.to_csv('Models/' + model + '/TrainArray.csv', index=False)
        self.TotalTesting.to_csv('Models/' + model + '/TestArray.csv', index=False)
        self.Totalrmse.to_csv('Models/' + model + '/RMSEArray.csv', index=False)
        print(model, 'arrays saved.')
        self.Totalrmse.drop(self.Totalrmse.index[0:], inplace=True)  # EMPTY RMSE ARRAY

        # If the p variable is equal to 1 then it saves predictions to csv.
        if p == 1:
            self.TotalPred.to_csv('Models/' + model + '/MatGenArray.csv', index=False)
            print(model, 'MatGen Array saved.')
        else:
            pass

        # print('Model:', model, 'Training Root Mean Squared Error:',
        #      np.sqrt(mean_squared_error(self.TotalTraining.iloc[0], self.TotalTraining.iloc[1])))
        # print('Model:', model, 'Testing Root Mean Squared Error:',
        #      np.sqrt(mean_squared_error(self.TotalTesting.iloc[0], self.TotalTesting.iloc[1])))


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


# This is the timer class that starts and stops timers for model comparison.
# Should change function structure eventually to automate this into GenModel maybe.

class Timer:

    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time

        self._start_time = None

        print(f"Elapsed time: {elapsed_time:0.4f} seconds")


t = Timer()

# This is the command portion of the script where you start timers, gen models, return arrays etc.
# You also establish model parameters here.

# RidgeCV
t.start()
GenModel(RidgeCV(), 1).ReturnArray("RidgeCV")
t.stop()

t.start()
GenModel(RidgeCV(), 1, p=1).ReturnArray("RidgeCV", p=1)
t.stop()

# OLS
t.start()
model_ols = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
GenModel(model_ols, 1).ReturnArray("OLS")
t.stop()

t.start()
GenModel(model_ols, 1, p=1).ReturnArray("OLS", p=1)
t.stop()

# KRR
t.start()
krr = KernelRidge(alpha=1.0)
GenModel(krr, 1).ReturnArray("KRR")
t.stop()

t.start()
GenModel(krr, 1, p=1).ReturnArray("KRR", p=1)
t.stop()

# SVM
t.start()
SVR = svm.SVR()
GenModel(SVR, 1).ReturnArray("SVR")
t.stop()

t.start()
GenModel(SVR, 1, p=1).ReturnArray("SVR", p=1)
t.stop()

# GPR
t.start()
gprkernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=gprkernel, random_state=0).fit(X, Y)
GenModel(gpr, 1).ReturnArray("GPR")
t.stop()

t.start()
GenModel(gpr, 1, p=1).ReturnArray("GPR", p=1)
t.stop()

# DTR
t.start()
dtr = tree.DecisionTreeRegressor(max_depth=7)
GenModel(dtr, 1).ReturnArray("DTR")
t.stop()

t.start()
GenModel(dtr, 1, p=1).ReturnArray("DTR", p=1)
t.stop()

# RFR
t.start()
rfr = RandomForestRegressor(max_depth=20)
GenModel(rfr, 1).ReturnArray("RFR")
t.stop()

t.start()
GenModel(rfr, 1, p=1).ReturnArray("RFR", p=1)
t.stop()

# MLPR
t.start()
mlpr = MLPRegressor(random_state=1, max_iter=500000, solver='lbfgs')
GenModel(mlpr, 1).ReturnArray("MLPR")
t.stop()

t.start()
GenModel(mlpr, 1, p=1).ReturnArray("MLPR", p=1)
t.stop()

# # To do:    statistics.
# graph of number of iterations vs accuracy
# fix decision tree depth, maybe relate to RFR depth
# fix perceptron, perhaps solver=lbfgs
# =IF(ABS(B159)<=0.16,"Within 0.16 of 0","Outside 0.16 of 0")