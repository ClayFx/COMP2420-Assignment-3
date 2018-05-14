import json
import os
import urllib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy.stats import ttest_ind, ttest_rel,ttest_1samp
from sklearn.preprocessing import scale

import itertools

plt.style.use('seaborn-notebook')
# inline figures
# %matplotlib inline

# just to make sure few warnings are not shown

stats = pd.read_hdf(os.path.join('data','baseball_team_stats_offensive_players.h5'))
print(stats.head(5))

playerLS = pd.read_hdf(os.path.join('data','baseball_players_offensive_stats.h5'))
print(playerLS.head(5))


# Q2.1
# Build a simple linear regression model to predict the number of wins for each entry in stats dataframe.
# Your features should be made up of the columns pertaining to normalized singles, double, triples, HR, and BB rates.
#
# Fit your model on data up to year 2002 and select the best performing model for data from 2003 to 2017.
# Use the fitted model to define a new [sabermetric](https://en.wikipedia.org/wiki/Sabermetrics) summary:
# which we'll call Offensive Predicted Wins (OPW). Also list the coefficients of your model


df_bf2002 = stats[stats['yearID'] <= 2002]
df_af2002 = stats[stats['yearID'] > 2002]
all_data = ['1B', '2B', '3B', 'HR', 'BB']
lm = LinearRegression(normalize=True)
y_before = df_bf2002.W
y_after = df_af2002.W
smallestK2 = 1000
bestOne = 0
combinations = list()
coef = list()

for i in range(1, 6, 1):
    combinations.extend(itertools.combinations(all_data, i))
for combi in combinations:
    x_before = df_bf2002[list(combi)]
    lm.fit(x_before, y_before)
    thisK2 = lm.score(df_af2002[list(combi)],y_after)
    if thisK2 < smallestK2:
        bestOne = combinations.index((combi))
        # lm.fit(x_before, y_before)
        smallestK2 = thisK2
        coef = lm.coef_
print(combinations[bestOne])
print('Coefficients are: ', coef)

# Q2.2
# Compute the OPW for each player based on the average rates in the `playerLS` DataFrame (5 marks)
#
# Notice that players essentially have the same features as teams, so you can use your model from Q2.1 to perform a prediction.
# Add this column to the playerLS DataFrame. Call\Name this colum OPW.

df_player = playerLS[['1B', '3B']]
# bestFeature = stats[['1B', '3B']]
# y_all = stats.W
bestFeature = df_bf2002[['1B', '3B']]
lm.fit(bestFeature, y_before)
opw_player = lm.predict(df_player)
playerLS['OPW'] = opw_player
print(playerLS.head())






















