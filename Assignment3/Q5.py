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

import math
plt.style.use('seaborn-notebook')


df = pd.DataFrame({
    'Temperature':[25,29,26,26,27,28,25,29,28,28,26,27],
    'Cloudy':['No','No','No','No','No','No','No','Yes','No','Yes','No','Yes'],
    'UV Index':['Low','Low','Low','Medium','Medium','High','High','Low','Medium','Medium','Low','Low'],
    'Humidity':['Low','High','Medium','Medium','High','High','Low','Low','High','High','Low','High'],
    'Rain':['No','No','No','No','No','No','No','Yes','Yes','Yes','Yes','Yes']
})
print(df)

# Q5.1
# What is the initial entropy of Rain?
entropy_rain = 5/12 * math.log(12/5, 2) + 7/12 * math.log(12/7, 2)
print(entropy_rain)

# Q5.2
# Which attribute would the decision-tree building algorithm choose at the root of the tree? (2 marks)
# Choose one through inspection and explain your reasoning in a sentence.
# Humidity. The reason is that raining will make the air rich of moisture so that the humidity will rise.

# Q 5.3
# Calculate and specify the information gain of the attribute you chose to split on in the previous question.
IG_rain_cloudy = 0.40672
IG_rain_humidity = 0.14654
IG_rain_UV = 0.14654
IG_rain_temp = 0.18739


# Q 5.4
# Consider a decision tree built from an arbitrary set of data.
# If the output is binary, what is the maximum training set error for this dataset? Explain your answer.
# (Please note that this is the error on the same dataset the tree was trained on.
#  A new test set could have arbitrary errors.)

# the number of roots should be as small as possible
# and the chosen root should make the IG as small as possible, i.e. the entropy of this root should be large.