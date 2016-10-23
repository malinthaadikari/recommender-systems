#!/usr/bin/python
from __future__ import division
from pandas import *

number_of_users = 943
number_of_movies = 1682
selected_recommendations = 10


def evaluate(weight, user_item_ratings,ratings_test):

    predicted = user_item_ratings * weight
    predicted = np.matrix.round(predicted)

    # Selecting top-N recommendations list
    # N=10
    top_n_ratings = np.zeros(shape=(number_of_users, selected_recommendations))
    temp = np.squeeze(np.asarray(predicted))
    for (b, m), value in np.ndenumerate(temp):
        top_n_ratings[b] = np.argsort(temp[b])[-selected_recommendations:][::-1]

    # Calculating HR
    test_ratings = {k: g['movie_id'].tolist() for k, g in ratings_test.groupby('user_id')}
    user_count = len(test_ratings)
    hit_count = 0
    ARHR_count = 0
    for key, value in test_ratings.iteritems():
        found = False
        for movie in value:
            if found:
                break
            if movie - 1 in top_n_ratings[key - 1]:
                hit_count += 1
                ARHR_count += 1/ (np.where(top_n_ratings[key-1]==(movie-1))[0][0] + 1)
                found = True
    print(hit_count / user_count)
    print(ARHR_count / user_count)

