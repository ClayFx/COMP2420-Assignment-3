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

import random

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
biggestR2 = 0
bestOne = 0
combinations = list()
coef = list()

for i in range(1, 6, 1):
    combinations.extend(itertools.combinations(all_data, i))
for combi in combinations:
    x_before = df_bf2002[list(combi)]
    lm.fit(x_before, y_before)
    thisR2 = lm.score(df_af2002[list(combi)],y_after)
    if thisR2 > biggestR2:
        bestOne = combinations.index((combi))
        # lm.fit(x_before, y_before)
        biggestR2 = thisR2
        coef = lm.coef_
print(combinations[bestOne])
print('Coefficients are: ', coef)

# Q2.2
# Compute the OPW for each player based on the average rates in the `playerLS` DataFrame (5 marks)
# Notice that players essentially have the same features as teams, so you can use your model from Q2.1 to perform a prediction.
# Add this column to the playerLS DataFrame. Call\Name this colum OPW.

df_player = playerLS[['1B', '2B', 'HR', 'BB']]
# bestFeature = stats[['1B', '3B']]
# y_all = stats.W
bestFeature = df_bf2002[['1B', '2B', 'HR', 'BB']]
lm.fit(bestFeature, y_before)
opw_player = lm.predict(df_player)
playerLS['OPW'] = opw_player
print(playerLS.head())

# Q2.3
#Plot and describe the relationship between the median salary (in millions) and the predicted number of wins for a player.
#Player should be active in the seasons between 2010 and 2012 inclusive, and should have an experience of at least 5 years.
df_player = playerLS[(playerLS.minYear <= 2010) & (playerLS.maxYear >= 2012) & ((playerLS.maxYear - playerLS.minYear) >= 5)]
plt.figure(figsize=(10,8))
ax = plt.subplot(111)
salary_in_million = df_player.salary / 1000000
ax.scatter(salary_in_million, df_player.OPW)
ax.set_xlabel("Median Salary(in millions)", fontsize=22)
ax.set_ylabel("OPW",fontsize=22)
ax.set_title("Relationship between the median salary and the OPW for a player",fontsize=24)
ax.set_xticklabels(ax.get_xticks(),fontsize=14)
ax.set_yticklabels(ax.get_yticks(),fontsize=14)
k, b = np.polyfit(salary_in_million, df_player.OPW, 1)
ax.plot(salary_in_million, k*salary_in_million + b, '-')
sns.despine()
plt.show()

# Q2.4

# YOUR CODE HERE
# create cost-effective column
# calculate by salary/opw
# which means how much it cost for each win
d = playerLS.copy(deep=True)
d['ce']=d['salary']/d['OPW']
# number of item to take
# take the  'numberofloop' most cost-effective item for iteration later
numberofloop=4
poss=['C','1B','2B','3B','SS','OF']
# tlist store index and salary as tuple
# tdict store index as key and (salary,OPW) as value
tdict={}
tlist=[]
counter=0
# initialized tdict and tlist
for p in poss:
    tlist.append([])
    if p!='OF':
        for i in range(1,numberofloop+1):
            tlist[counter].append((d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).index.values[0],d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).salary.values[0]))
            tdict[d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).index.values[0]]=(d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).salary.values[0],d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).OPW.values[0])
    else:
        for i in range(1,numberofloop*4+1):
            tlist[counter].append((d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).index.values[0],d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).salary.values[0]))
            tdict[d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).index.values[0]]=(d[(d['POS']==p)&(d.index.isin(tdict.keys())==False)].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).salary.values[0],d[d['POS']==p].sort_values('OPW').tail(15).sort_values('ce').head(i).tail(1).OPW.values[0])
    counter=counter+1
# initialized 2 combine list, first use for first 5 position and second use for position OF
combine1=list(itertools.product(tlist[0],tlist[1],tlist[2],tlist[3],tlist[4]))
combine2=list(itertools.combinations(tlist[5],4))
# comparator to find the max OPW
maxwin=0
# will store the index of the maxwin as list
maxcombine=None
limit=25000000.0
# loop through every possible combinations
for q in combine1:
    for w in combine2:
        wins=sum([pair[1] for pair in q+w])
        if (wins<limit) & (wins>maxwin):
            maxwin=wins
            maxcombine=list(q+w)
print("After first iteration get total salary of: ",str(sum([pair[1] for pair in maxcombine])))
print("After first method get total OPW:",str(d[d.index.isin([pair[0] for pair in maxcombine])]['OPW'].sum()))
print("After first method get average OPW:",str(d[d.index.isin([pair[0] for pair in maxcombine])]['OPW'].sum()/9))
for i in range(1,50):
    r=random.randint(0,8)
    pos=d[d.index==maxcombine[r][0]]['POS'].values[0]
    salary=limit-sum([q[1] for q in maxcombine])+maxcombine[r][1]
    maxcombine.pop(r)
    maxcombine.append((d[(d['POS']==pos)&(d['salary']<=salary)&(d.index.isin([pair[0] for pair in maxcombine])==False)].sort_values('OPW').tail(1).index.values[0],d[(d['POS']==pos)&(d['salary']<=salary)&(d.index.isin([pair[0] for pair in maxcombine])==False)].sort_values('OPW').tail(1).salary.values[0]))

print("After LNS get total salary of :",str(sum([pair[1] for pair in maxcombine])))
print("After LNS get total OPW of :",str(d[d.index.isin([pair[0] for pair in maxcombine])]['OPW'].sum()))
print("After LNS get average OPW of :",str(d[d.index.isin([pair[0] for pair in maxcombine])]['OPW'].sum()/9))
print(d[d.index.isin([pair[0] for pair in maxcombine])][['playerID','salary','POS','OPW']])
# for i in range(0,len(combine))



















