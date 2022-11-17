import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

modelset = ('DTR', 'GPR', 'KRR', 'MLPR', 'OLS', 'RFR', 'RidgeCV', 'SVR')


# Function portion of code. These are built to automate features.

def plotdata(model):
    testdata = pd.read_csv('Models/' + model + '/TestArray.csv')
    traindata = pd.read_csv('Models/' + model + '/TrainArray.csv')
    fig, ax = plt.subplots()

    ax.scatter([testdata.iloc[:, 0]], [testdata.iloc[:, 1]], color='blue', marker='o', edgecolors='white', alpha=0.8)
    ax.scatter([traindata.iloc[:, 0]], [traindata.iloc[:, 1]], color='green', marker='s', edgecolors='white', alpha=0.8)
    ax.legend(['Testing Data', 'Training Data'])
    plt.axis([-1.5, 0.5, -1.5, 0.5])
    plt.title(model + " Pred ΔG vs Actual ΔG", pad=15)
    plt.xlabel("Predicted ΔG")
    plt.ylabel("Actual ΔG")
    line = mlines.Line2D([0, 1], [0, 1], color='red', linestyle='dashed')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    circlehighlight = plt.Circle((0, 0), 0.15, color='r', fill=False)
    plt.gca().add_patch(circlehighlight)
    plt.savefig('Models/' + model + '/TrainVSTest.png')
    plt.show()


def plotrmse(model):
    rmse = pd.read_csv('Models/' + model + '/RMSEArray.csv')
    plt.bar([rmse.columns[0]], [rmse.iloc[0, 0]])
    plt.bar([rmse.columns[1]], [rmse.iloc[1, 0]])
    plt.ylim(0, 0.5)
    plt.title(model + " Training RMSE vs Testing RMSE", pad=15)
    plt.ylabel("Root Mean Squared Error")
    plt.savefig('Models/' + model + '/RMSEBar.png')
    plt.show()


def autormse():
    for i, modelstring in enumerate(modelset):
        rmse = pd.read_csv('Models/' + modelstring + '/RMSEArray.csv')
        plt.bar([rmse.columns[0]], [rmse.iloc[0, 0]])
        plt.bar([rmse.columns[1]], [rmse.iloc[1, 0]])
        continue
    plt.ylim(0, 0.5)
    plt.title("Training RMSE vs Testing RMSE", pad=15)
    plt.ylabel("Root Mean Squared Error")
    plt.savefig('Models/' + modelstring + '/RMSEBar.png')
    plt.show()


def plotpred(model):
    preddata = pd.read_csv('Models/' + model + '/MatGenArray.csv')
    fig, ax = plt.subplots()
    x = preddata.iloc[:, 0]
    y = preddata.iloc[:, 1]
    # ax.scatter([preddata.iloc[:, 0]], [preddata.iloc[:, 1]], color='blue', edgecolors='white', alpha=0.8)
    ax.plot(x, y)
    plt.xlabel("Material Number")
    plt.ylabel("Predicted ΔG")
    plt.title(model + " Material Number vs Predicted ΔG", pad=15)
    ax.add_patch(Rectangle(xy=(0, -0.15), width=300, height=0.30, linewidth=1, color='red', fill=False))
    plt.show()


# Calling functions portion of code. This should eventually be automated to request a model list
# and iterate through that list with the functions you'd like.

plotdata('DTR')
plotdata('GPR')
plotdata('KRR')
plotdata('MLPR')
plotdata('OLS')
plotdata('RFR')
plotdata('RidgeCV')
plotdata('SVR')

plotrmse('DTR')
plotrmse('GPR')
plotrmse('KRR')
plotrmse('MLPR')
plotrmse('OLS')
plotrmse('RFR')
plotrmse('RidgeCV')
plotdata('SVR')

plotpred('RFR')
