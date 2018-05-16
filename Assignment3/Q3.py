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


from sklearn.metrics import make_scorer, accuracy_score

plt.style.use('seaborn-notebook')

df_credit = pd.read_hdf(os.path.join('data','df_credit.h5'))
df_credit.head()

# Q 3.1
# Write a **ten-fold cross validation** to estimate the optimal value for $k$ for the data set.
# You need to consider only values between 20 to 50(inclusive) for $k$
performance = []
X_credit = df_credit.drop(columns=['DEFAULT'])
Y_credit = df_credit.DEFAULT
X_train, X_test, Y_train, Y_test = train_test_split(X_credit, Y_credit, test_size=0.33, random_state = 3)
k_set = [i for i in range(20, 51)]
for k in k_set:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_model, X_train, Y_train, cv=10,scoring=make_scorer(accuracy_score))
    performance.append(np.mean(score))
    print(k)
print(performance)
best_k = k_set[np.argmax(performance)]
print(best_k)


