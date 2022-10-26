import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

TrainData = pd.read_csv('TrainData.csv')

X = TrainData.values[:, [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
Y = TrainData.values[:, 23]
m = len(Y)

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
