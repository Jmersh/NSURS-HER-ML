import numpy as np
import pandas as pd
import sklearn
from IPython.core.display_functions import display
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
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

TrainData = pd.read_csv('TrainData.csv')

X = TrainData.values[:, [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
Y = TrainData.values[:, 23]
m = len(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

print('Total no of training examples (m) = %s \n' % (m))

model_ols = linear_model.LinearRegression(normalize=True)
model_ols.fit(X, Y)

print('OLS Model Complete, listing stats')
print('coef= ', model_ols.coef_)
print('intercept= ', model_ols.intercept_)

olsprede = pd.DataFrame(model_ols.predict(X), columns=['Predicted E'])
olsacte = pd.DataFrame(Y, columns=['Actual E'])
olsacte = olsacte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new dataframe
df_actual_vs_predicted = pd.concat([olsacte, olsprede], axis=1)
df_actual_vs_predicted.T

mse = mean_squared_error(olsprede, olsacte)
rmse = np.sqrt(mse)
print('OLS root mean squared error:', rmse)

krr = KernelRidge(alpha=1.0)
krr.fit(X, Y)

print('KRR Model Complete')

krrprede = pd.DataFrame(krr.predict(X), columns=['Predicted E'])
krracte = pd.DataFrame(Y, columns=['Actual E'])
krracte = krracte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new dataframe
krr_actual_vs_predicted = pd.concat([krracte, krrprede], axis=1)
krr_actual_vs_predicted.T
krrrmse = np.sqrt(mean_squared_error(krrprede, krracte))

print('KRR root mean squared error:', krrrmse)

regr = svm.SVR()
regr.fit(X, Y)

print('SVR Model Complete')

svrprede = pd.DataFrame(regr.predict(X), columns=['Predicted E'])
svracte = pd.DataFrame(Y, columns=['Actual E'])
svracte = krracte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new dataframe
svr_actual_vs_predicted = pd.concat([svracte, svrprede], axis=1)
svr_actual_vs_predicted.T
svrrmse = np.sqrt(mean_squared_error(svrprede, svracte))

print('SVR root mean squared error:', svrrmse)

gprkernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=gprkernel, random_state=0).fit(X, Y)
gpr.score(X, Y)

print('GPR Model Complete, score:', gpr.score(X, Y))

gprprede = pd.DataFrame(gpr.predict(X), columns=['Predicted E'])
gpracte = pd.DataFrame(Y, columns=['Actual E'])
gpracte = gpracte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new dataframe
gpr_actual_vs_predicted = pd.concat([gpracte, gprprede], axis=1)
gpr_actual_vs_predicted.T
gprrmse = np.sqrt(mean_squared_error(gprprede, gpracte))

print('GPR root mean squared error:', gprrmse)

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, Y)

print('DTR Model Complete')

clfprede = pd.DataFrame(clf.predict(X), columns=['Predicted E'])
clfacte = pd.DataFrame(Y, columns=['Actual E'])
clfacte = gpracte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new dataframe
clf_actual_vs_predicted = pd.concat([clfacte, clfprede], axis=1)
clf_actual_vs_predicted.T
clfrmse = np.sqrt(mean_squared_error(clfprede, clfacte))

print('DTR root mean squared error:', clfrmse)

rtr = RandomForestRegressor()
rtr = rtr.fit(X, Y)

print('RTR Model Complete')

rtrprede = pd.DataFrame(rtr.predict(X), columns=['Predicted E'])
rtracte = pd.DataFrame(Y, columns=['Actual E'])
rtracte = rtracte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new dataframe
rtr_actual_vs_predicted = pd.concat([rtracte, rtrprede], axis=1)
rtr_actual_vs_predicted.T
rtrrmse = np.sqrt(mean_squared_error(rtrprede, rtracte))

print('RTR root mean squared error:', rtrrmse)

rcv = RidgeCV()
rcv = rcv.fit(X, Y)

print('RCV Model Complete')

rcvprede = pd.DataFrame(rcv.predict(X), columns=['Predicted E'])
rcvacte = pd.DataFrame(Y, columns=['Actual E'])
rcvacte = rcvacte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new dataframe
rcv_actual_vs_predicted = pd.concat([rcvacte, rcvprede], axis=1)
rcv_actual_vs_predicted.T
rcvrmse = np.sqrt(mean_squared_error(rcvprede, rcvacte))

print('RCV root mean squared error:', rcvrmse)

mlpr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, Y_train)

print('MLPR Model Complete, score:', mlpr.score(X_test, Y_test))

mlprprede = pd.DataFrame(mlpr.predict(X), columns=['Predicted E'])
mlpracte = pd.DataFrame(Y, columns=['Actual E'])
mlpracte = rtracte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new dataframe
mlpr_actual_vs_predicted = pd.concat([mlpracte, mlprprede], axis=1)
mlpr_actual_vs_predicted.T
mlprrmse = np.sqrt(mean_squared_error(mlprprede, mlpracte))

print('MLPR root mean squared error:', mlprrmse)


class GenModel:
    TotalTraining = pd.DataFrame(columns=['Tr Predicted E', 'Tr Actual E'])
    TotalTesting = pd.DataFrame(columns=['Te Predicted E', 'Te Actual E'])
    Totalrmse = pd.DataFrame(columns=['Training RMSE', 'Testing RMSE'])

    def __init__(self, model):
        for ModelN in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=ModelN)
            model.fit(X_train, Y_train)

            # Training Array Maker
            TrGMprede = pd.DataFrame(model.predict(X_train), columns=['Tr Predicted E'])
            TrGMacte = pd.DataFrame(Y_train, columns=['Tr Actual E'])
            TrGMacte = TrGMacte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new
            # dataframe
            TrGM_actual_vs_predicted = pd.concat([TrGMacte, TrGMprede], axis=1)
            self.TotalTraining = pd.concat([self.TotalTraining, TrGM_actual_vs_predicted], axis=0)
            TrGMrmse = np.sqrt(mean_squared_error(TrGMprede, TrGMacte))

            # Test Array Maker
            TeGMprede = pd.DataFrame(model.predict(X_test), columns=['Te Predicted E'])
            TeGMacte = pd.DataFrame(Y_test, columns=['Te Actual E'])
            TeGMacte = TeGMacte.reset_index(drop=True)  # Drop the index so that we can concat it, to create new
            # dataframe
            TeGM_actual_vs_predicted = pd.concat([TeGMacte, TeGMprede], axis=1)
            self.TotalTesting = pd.concat([self.TotalTesting, TeGM_actual_vs_predicted], axis=0)
            TeGMrmse = np.sqrt(mean_squared_error(TeGMprede, TeGMacte))

            self.Totalrmse.loc[len(self.Totalrmse.index)] = [TrGMrmse, TeGMrmse]
            continue

    def ReturnArray(self, model):
        # printing arrays for debug
        # print(self.TotalTraining)
        # print(self.TotalTesting)

        self.Totalrmse.loc['mean'] = self.Totalrmse.mean()
        # print(self.Totalrmse)

        # print model completion
        print(model, 'generation complete.')
        # saving arrays for future use
        self.TotalTraining.to_csv('Models/' + model + '/TrainArray.csv', index=False)
        self.TotalTesting.to_csv('Models/' + model + '/TestArray.csv', index=False)
        self.Totalrmse.to_csv('Models/' + model + '/RMSEArray.csv', index=False)
        print(model, 'arrays saved.')


        # print('Model:', model, 'Training Root Mean Squared Error:',
        #      np.sqrt(mean_squared_error(self.TotalTraining.iloc[0], self.TotalTraining.iloc[1])))
        # print('Model:', model, 'Testing Root Mean Squared Error:',
        #      np.sqrt(mean_squared_error(self.TotalTesting.iloc[0], self.TotalTesting.iloc[1])))


GenModel(RidgeCV()).ReturnArray("RidgeCV")


# # To do: loop training and prediction 10 times, redo LearningModelGel with classes, add prediction and use split
# training data, graphs, predict our MatGenOutput and create array or graph, statistics.
# graph of number of iterations vs accuracy
