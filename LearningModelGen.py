import numpy as np
import pandas as pd
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

TrainData = pd.read_csv('TrainData.csv')

X = TrainData.values[:, [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
Y = TrainData.values[:, 23]
m = len(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=1)

print('Total no of training examples (m) = %s \n' % (m))

model_ols = linear_model.LinearRegression(normalize=True)
model_ols.fit(X, Y)

print('OLS Model Complete, listing stats')
print('coef= ', model_ols.coef_)
print('intercept= ', model_ols.intercept_)

olsprede = pd.DataFrame(model_ols.predict(X), columns=['Predicted E'])
olsacte = pd.DataFrame(Y, columns=['Actual E'])
olsacte = olsacte.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
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
krracte = krracte.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
krr_actual_vs_predicted = pd.concat([krracte, krrprede], axis=1)
krr_actual_vs_predicted.T
krrrmse = np.sqrt(mean_squared_error(krrprede, krracte))

print('KRR root mean squared error:', krrrmse)

regr =svm.SVR()
regr.fit(X, Y)

print('KRR Model Complete')

svrprede = pd.DataFrame(regr.predict(X), columns=['Predicted E'])
svracte = pd.DataFrame(Y, columns=['Actual E'])
svracte = krracte.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
svr_actual_vs_predicted = pd.concat([svracte, svrprede], axis=1)
svr_actual_vs_predicted.T
svrrmse = np.sqrt(mean_squared_error(svrprede, svracte))

print('SVR root mean squared error:', svrrmse)


gprkernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=gprkernel, random_state=0).fit(X, Y)
gpr.score(X,Y)

print('GPR Model Complete, score:', gpr.score(X, Y))

gprprede = pd.DataFrame(gpr.predict(X), columns=['Predicted E'])
gpracte = pd.DataFrame(Y, columns=['Actual E'])
gpracte = gpracte.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
gpr_actual_vs_predicted = pd.concat([gpracte, gprprede], axis=1)
gpr_actual_vs_predicted.T
gprrmse = np.sqrt(mean_squared_error(gprprede, gpracte))

print('GPR root mean squared error:', gprrmse)

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,Y)

print('DTR Model Complete')

clfprede = pd.DataFrame(clf.predict(X), columns=['Predicted E'])
clfacte = pd.DataFrame(Y, columns=['Actual E'])
clfacte = gpracte.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
clf_actual_vs_predicted = pd.concat([clfacte, clfprede], axis=1)
clf_actual_vs_predicted.T
clfrmse = np.sqrt(mean_squared_error(clfprede, clfacte))

print('DTR root mean squared error:', clfrmse)

rtr = RandomForestRegressor()
rtr = rtr.fit(X,Y)

print('RTR Model Complete')

rtrprede = pd.DataFrame(rtr.predict(X), columns=['Predicted E'])
rtracte = pd.DataFrame(Y, columns=['Actual E'])
rtracte = rtracte.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
rtr_actual_vs_predicted = pd.concat([rtracte, rtrprede], axis=1)
rtr_actual_vs_predicted.T
rtrrmse = np.sqrt(mean_squared_error(rtrprede, rtracte))

print('RTR root mean squared error:', rtrrmse)

mlpr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, Y_train)

print('MLPR Model Complete, score:', mlpr.score(X_test, Y_test))

mlprprede = pd.DataFrame(mlpr.predict(X), columns=['Predicted E'])
mlpracte = pd.DataFrame(Y, columns=['Actual E'])
mlpracte = rtracte.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
mlpr_actual_vs_predicted = pd.concat([mlpracte, mlprprede], axis=1)
mlpr_actual_vs_predicted.T
mlprrmse = np.sqrt(mean_squared_error(mlprprede, mlpracte))

print('MLPR root mean squared error:', mlprrmse)
