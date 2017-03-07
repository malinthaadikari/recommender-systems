import pandas as pd
import numpy as np
from scipy import spatial

def data_preprocess(data_file):
    # here we define column name in our data file
    r_cols = ['user_id', 'movie_id', 'rating','timestamp']
    # loading data in our raw data file into pandas dataframe
    user_ratings = pd.read_csv(data_file, sep='\s+', names=r_cols)
    # removing timestamp column as it is not needed in this stage
    user_ratings.__delitem__('timestamp')
    # here we aggregate using a similar function like groupby
    # here we use mean function as the aggregate funtion. One important thing is reindexing. You can try with and without
    # reindexing. Using reindexing adds column for never rated movie.
    user_ratings_matrix = user_ratings.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0,
                                                   aggfunc=np.mean).reindex(columns=np.arange(1,6), fill_value=0)
    # now we have user-movie ratings matrix

    # Creates a list containing 5 lists, each of 8 items, all set to 0
    w, h = 4, 4;
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for index, row in user_ratings_matrix.iterrows():
        for index_internal, row_internal in user_ratings_matrix.iterrows():
            Matrix[index][index_internal] = 1 - spatial.distance.cosine(row.as_matrix(), row_internal.as_matrix())
            print Matrix[index][index_internal]

    # now we got similarity matrix
    #go through each user and find the most similar user for that user
    return user_ratings_matrix