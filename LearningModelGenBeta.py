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

TrainData = pd.read_csv('TrainData.csv', index_col=[0])

X = TrainData.iloc[:, [1,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
Y = TrainData.values[:, 22]


# X = TrainData.values[:, [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
# Y = TrainData.values[:, 23]
m = len(Y)

print('Total no of training examples (m) = %s \n' % (m))


class GenModel:
    TotalTraining = pd.DataFrame(columns=['Tr Predicted E', 'Tr Actual E'])
    TotalTesting = pd.DataFrame(columns=['Te Predicted E', 'Te Actual E'])
    Totalrmse = pd.DataFrame(columns=['Training RMSE', 'Testing RMSE'])
    # TotalIndice = pd.DataFrame(columns=['Name'])

    def __init__(self, model, n=10):
        for ModelN in range(n):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=ModelN)
            # Indice = pd.DataFrame([X_train.index.values, X_test.index.values, Y_train.index.values, Y_test.index.values], columns=['X_train Ind' + str(ModelN), 'X_test Ind' + str(ModelN), 'Y_train Ind' + str(ModelN), 'Y_test Ind' + str(ModelN)])
            # self.TotalIndice = pd.concat([self.TotalIndice, Indice], axis=1)
            scaler = StandardScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_test = scaler.fit_transform(X_test)
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
            #TeGMacte = TeGMacte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new
            # dataframe
            TeGM_actual_vs_predicted = pd.concat([TeGMacte, TeGMprede], axis=1)
            self.TotalTesting = pd.concat([self.TotalTesting, TeGM_actual_vs_predicted], axis=1)
            TeGMrmse = np.sqrt(mean_squared_error(TeGMprede, TeGMacte))

            self.Totalrmse.loc[len(self.Totalrmse.index)] = [TrGMrmse, TeGMrmse]
            continue

    def ReturnArray(self, model):
        self.Totalrmse.loc['mean'] = self.Totalrmse.mean()
        self.TotalTesting.dropna(axis='columns', inplace=True)
        self.TotalTraining.dropna(axis='columns', inplace=True)
        # printing arrays for debug
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

        # print('Model:', model, 'Training Root Mean Squared Error:',
        #      np.sqrt(mean_squared_error(self.TotalTraining.iloc[0], self.TotalTraining.iloc[1])))
        # print('Model:', model, 'Testing Root Mean Squared Error:',
        #      np.sqrt(mean_squared_error(self.TotalTesting.iloc[0], self.TotalTesting.iloc[1])))


GenModel(RidgeCV(), 1).ReturnArray("RidgeCV")

model_ols = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
GenModel(model_ols, 1).ReturnArray("OLS")

krr = KernelRidge(alpha=1.0)
GenModel(krr, 1).ReturnArray("KRR")

SVR = svm.SVR()
GenModel(SVR, 1).ReturnArray("SVR")

gprkernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=gprkernel, random_state=0).fit(X, Y)
GenModel(gpr, 1).ReturnArray("GPR")

dtr = tree.DecisionTreeRegressor()
GenModel(dtr, 1).ReturnArray("DTR")

rfr = RandomForestRegressor()
GenModel(rfr, 1).ReturnArray("RFR")

mlpr = MLPRegressor(random_state=1, max_iter=500)
GenModel(mlpr, 1).ReturnArray("MLPR")

# # To do:  graphs, predict our MatGenOutput and create array or graph, statistics.
# graph of number of iterations vs accuracy
