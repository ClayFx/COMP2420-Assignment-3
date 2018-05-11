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
import warnings
warnings.filterwarnings("ignore")

df_tweets = pd.read_hdf(os.path.join('data','yt_tweets_df.h5'))
print(df_tweets.head(1))

# Q1.1
# Compare the mean for '#friends' for tweets in language 'en' (lang_tweet='en') against
# the overall mean value, 612. (5 marks)
tweet_en = df_tweets[df_tweets.lang_tweet == 'en']
friends_en = tweet_en['#friends']
print(ttest_1samp(friends_en,612))


#Q1.2
# Compare the mean for '#friends' for tweets tagged with language English (lang_tweet='en') against
# the tweets tagged with language Japanese (lang_tweet='ja'). (5 marks)
tweet_ja = df_tweets[df_tweets.lang_tweet == 'ja']
friends_ja = tweet_ja['#friends']
print(ttest_1samp(friends_ja,friends_en.mean()))


# Q 1.3
# Compare the mean for '#followers' against '#friends 'for tweets tagged with language English (lang_tweet='en').
mean_follower = df_tweets['#followers'].mean()
print(ttest_1samp(friends_en,mean_follower))






























