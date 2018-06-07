import pandas as pd
import numpy as np
from numpy import inf
from sklearn.model_selection import train_test_split
import sklearn
import warnings
warnings.filterwarnings('ignore')

# importing dataset
db = pd.read_csv('adult.csv')

# lets categorise our target 0 - for <=50 and 1 - for >50
db.loc[db['income']=='<=50K', 'income'] = 0
db.loc[db['income']=='>50K', 'income'] = 1
db['income'] = pd.to_numeric(db['income'])

# splitting data on train-test
train_base, test_base = train_test_split(db, test_size=0.2)

# in our data set we have categorical features and continuous features

# let's log-scale our continuous features for train and test separately

continuous_feature_list = ['capital-gain', 'capital-loss', 'hours-per-week']
continuous_feature_list_mean = []
for feature in continuous_feature_list:
    train_base[feature+'_mean'] = np.log(train_base[feature]).replace(-inf,0)
    train_base[feature+'_mean'] = train_base[feature+'_mean']/train_base[feature+'_mean'].max()
    test_base[feature + '_mean'] = np.log(test_base[feature]).replace(-inf, 0)
    test_base[feature + '_mean'] = test_base[feature + '_mean'] / test_base[feature + '_mean'].max()
    # list for remembering names of our modified continuous features
    continuous_feature_list_mean += [feature+'_mean']

# for categorical features I sugget to apply mean-encoding, where we transform each category into it's target-mean
# representation. Of course, to avoid overfittig we should use only mean encodings from train dataset
# we have to transfer encodings from train to test and NOT to calculate them separately on test dataset.
for feature in train_base.drop(['income', 'fnlwgt']+continuous_feature_list + continuous_feature_list_mean,axis=1).columns:
    db_group = train_base.groupby(feature)['income'].mean()
    train_base[feature+'_mean'] = train_base[feature].replace(list(db_group.index.values), list(db_group.values))
    test_base[feature + '_mean'] = test_base[feature].replace(list(db_group.index.values), list(db_group.values))

# datasets cleaning
train_x = train_base.dropna().loc[:, train_base.columns.str.contains('mean')]
train_y = train_base.dropna().loc[:, 'income']

test_x = test_base.dropna().loc[:, test_base.columns.str.contains('mean')]
test_y = test_base.dropna().loc[:, 'income']

# calling training lib - LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(train_x, train_y)

# printing results for train and test
print('Train score = '+str(model.score(train_x, train_y)))
print('Test score = ' + str(model.score(test_x, test_y)))
