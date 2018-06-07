import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

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




df.drop(df.columns[[3,4,7]], axis=1, inplace=True)
ratings.drop( "unix_timestamp", inplace = True, axis = 1 )
movies.drop(movies.columns[[3,4]], inplace = True, axis = 1 )


#Pivot Table(This creates a matrix of users and movie_ratings)
ratings_matrix = ratings.pivot_table(index=['movie_id'],columns=['user_id'],values='rating').reset_index(drop=True)
ratings_matrix.fillna( 0, inplace = True )

#Cosine Similarity(Creates a cosine matrix of similaraties ..... which is the pairwise distances
# between two items )

movie_similarity = 1 - pairwise_distances( ratings_matrix.as_matrix(), metric="cosine" )
np.fill_diagonal( movie_similarity, 0 )
ratings_matrix = pd.DataFrame( movie_similarity )