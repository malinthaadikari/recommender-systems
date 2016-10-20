import pandas as pand
from pandas import *

user_ratings = np.zeros(shape=(943, 1682))

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pand.read_csv('resources/data/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pand.read_csv('resources/data/ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pand.read_csv('resources/data/ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

ratings_base = pand.read_csv('resources/data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pand.read_csv('resources/data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
# print(ratings_base.shape, ratings_test.shape)
ratings.__delitem__('unix_timestamp')
numpyMatrix = ratings.as_matrix()

for index, row in ratings_base.iterrows():
   user_ratings[row[0]-1][row[1]-1] = row[2]
print(user_ratings.T*user_ratings)