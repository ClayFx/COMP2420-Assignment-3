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























