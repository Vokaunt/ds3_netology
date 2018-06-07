import numpy as np
import pandas as pd

#Loading movielens data

#User's data
tag_cols = ['user_id', 'del1', 'movie_id', 'del2', 'tag', 'del3', 'timestamp']
tags = pd.read_csv('ml-100k/tags.dat', sep=':', names=tag_cols, parse_dates=True)
tags.drop(tags.columns[[1,3,5]], inplace = True, axis = 1)
#Ratings
rating_cols = ['user_id','del1', 'movie_id', 'del2', 'rating', 'del3', 'timestamp']
ratings = pd.read_csv('ml-100k/ratings.dat', sep=':', names=rating_cols)
ratings.drop(ratings.columns[[1,3,5]], inplace = True, axis = 1)
#Movies
movie_cols = ['movie_id', 'del1', 'title', 'del2', 'genres']
movies = pd.read_csv('ml-100k/movies.dat', sep=':', names=movie_cols, usecols=range(5),encoding='latin-1')
movies.drop(movies.columns[[1,3]], inplace = True, axis = 1)


#Merging movie data with their ratings
movie_ratings = pd.merge(ratings, movies, how='left', left_on='movie_id', right_on='movie_id')
#merging movie_ratings data with the User's dataframe
df = pd.merge(movie_ratings, tags, how='left', left_on=['user_id', 'movie_id'], right_on=['user_id', 'movie_id'])
 #pre-processing
 #dropping colums that aren't needed
df.drop(df.columns[[3,7]], axis=1, inplace=True)

from sklearn.model_selection import train_test_split

train_base, test_base = train_test_split(df, test_size=0.2)
# for categorical features I sugget to apply mean-encoding, where we transform each category into it's target-mean
# representation. Of course, to avoid overfittig we should use only mean encodings from train dataset
# we have to transfer encodings from train to test and NOT to calculate them separately on test dataset.
for feature in train_base.drop(['rating','title'],axis=1).columns:
    db_group = train_base.groupby(feature, as_index=False)['rating'].mean()
    print(len(train_base))
    train_base = pd.merge(train_base,db_group, how='left', left_on=feature, right_on=feature, suffixes=('', '_mean'))
    train_base.columns.values[-1] = str(feature) + '_mean'
    print(len(train_base))
    print(len(test_base))
    test_base = pd.merge(test_base, db_group, how='left', left_on=feature, right_on=feature, suffixes=('', '_mean'))
    test_base.columns.values[-1] = str(feature) + '_mean'
    print(len(test_base))

# finding global mean rating (on train dataset) for filling NaN values
mean_rating = train_base['rating'].mean()
train_base = train_base.fillna(mean_rating)
test_base = test_base.fillna(mean_rating)

train_x = train_base.loc[:, train_base.columns.str.contains('mean')]
train_y = train_base.loc[:, 'rating']

test_x = test_base.loc[:, train_base.columns.str.contains('mean')]
test_y = test_base.loc[:, 'rating']

# calling training lib - LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
model = linear_model.LinearRegression()

model.fit(train_x, train_y)

train_pred = model.predict(train_x)
test_pred = model.predict(test_x)
# printing results for train and test
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Root mean squared error for train: %.2f"
      % np.sqrt(mean_squared_error(train_y, train_pred)))
print("Root mean squared error for test: %.2f"
      % np.sqrt(mean_squared_error(test_y, test_pred)))