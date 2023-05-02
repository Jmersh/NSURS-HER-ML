# NSURS-HER-ML

**UPDATE: As of 5/2/2023, this project is almost fully operational with minimal script changes needed**

## Purpose

Welcome to the NSURS-HER-ML project! This repository is part of an undergraduate project titled "Screening of Highly Active Hydrogen Evolution Reaction Catalysts: Comparative Analysis of Typical Machine Learning Methods". You can read my paper and check out my presentation on this GitHub as well.

## Features

The NSURS-HER-ML project provides several features to make it easy for you to generate materials data, import training data, build models, compare models, and predict materials. Some of the key features of this project are:

- Generate a CSV of materials with descriptors to later predict
- Train models and save outputs to CSV
- Generate statistics and graphs for models
- Import descriptors from materials database
- TPOT and Ludwig Auto-ML pipeline

## To-Do

We are continuously improving the NSURS-HER-ML project to make it more efficient and user-friendly. Some of the tasks we are currently working on include:

- Check association values between descriptors
- Optimize models
- Add more comments
- Add examples of what project can do

## Requirements

To use the NSURS-HER-ML project, you need to have the following software installed:

- [Python](https://www.python.org/downloads/)
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [SciPy](https://scipy.org/install/)
- [Scikit-Learn](https://scikit-learn.org/stable/install.html)
- [Pymatgen](https://pymatgen.org/)
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [TPOT](https://github.com/EpistasisLab/tpot)
- [Ludwig](https://ludwig-ai.github.io/ludwig-docs/getting_started/install/)
- [periodictable](https://github.com/pkienzle/periodictable)
- [PyTorch](https://pytorch.org/get-started/locally/)

You also need to have a Python IDE installed. Here are some popular options:

- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Simplilearn](https://www.simplilearn.com/tutorials/python-tutorial/python-ide)

## Abstract

Abstract
Hydrogen evolution reaction (HER) catalysts play a critical role in renewable energy technologies such as hydrogen production. Identifying novel catalysts with high efficiency and low cost is a significant challenge. In this study, we investigate the effectiveness of supervised machine learning techniques for screening HER catalysts. 

We applied various machine learning models to predict the probability of a list of 300 transition metal borides, carbides, and nitrides being a highly efficient HER catalyst. Elemental descriptors that can be easily obtained by the periodic table for Metal (M) and carbon, nitrogen, boron (X) were primarily used. Materials with published Gibbs free energy of hydrogen adsorption (ΔGH*) values were utilized for our training data. We compared the different machine learning models based on root mean square error (RMSE), speed of model generation, and speed of model prediction. 

Our results showed that the Random Forest Regression model produced the lowest testing RMSE and was chosen to examine the 300 materials, Out of which 29 materials were predicted to be high-performance HER catalysts. Our approach can efficiently identify promising HER catalysts with high accuracy, providing guidance for further experimental and theoretical investigations.


## Training Data

The training data for this project includes Gibbs Free Energy values and names of materials used. Other descriptors are generated from the project. The data was sourced from the following publication:

Sun, X.; Zheng, J.; Gao, Y.; Qiu, C.; Yan, Y.; Yao, Z.; Deng, S.; Wang, J. Machine-Learning-Accelerated Screening of Hydrogen Evolution Catalysts in MBenes Materials. Applied Surface Science 2020, 526, 146522. https://doi.org/10.1016/j.apsusc.2020.146522. (LICENSE DOES NOT APPLY TO THIS DATA)

## License

The NSURS-HER-ML project is licensed under the MIT License. This means that you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software. However, the license only applies to the software and associated documentation files. Any training data used in this project has its own separate license. Please refer to the source of the data for information on its license.

Copyright (c) 2022 Jordan Mershimer

