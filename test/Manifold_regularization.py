#!/usr/bin/python
from __future__ import division
import pandas as pand
from pandas import *

number_of_users = 943
number_of_movies = 1682
selected_recommendations = 10

def evaluate(weight):
    predicted = user_ratings * weight
    predicted = np.matrix.round(predicted)

    # Selecting top-N recommendations list
    # N=10
    import bottleneck as bottleneck
    top_N_ratings = np.zeros(shape=(number_of_users, selected_recommendations))
    temp = np.squeeze(np.asarray(predicted))
    for (b, m), value in np.ndenumerate(temp):
        top_N_ratings[b] = np.array(
            bottleneck.argpartsort(-temp[b], selected_recommendations)[:selected_recommendations])

    # Calculating HR

    test_ratings = {k: g['movie_id'].tolist() for k, g in ratings_test.groupby('user_id')}
    user_count = len(test_ratings)
    hit_count = 0

    for key, value in test_ratings.iteritems():
        found = False
        for movie in value:
            if found:
                break
            if movie - 1 in top_N_ratings[key - 1]:
                hit_count = hit_count + 1
                found = True

    print(hit_count / user_count)


def LapLRSFunc(X):
    item_count = X.shape[1]

    # creating Identity matrix
    identity_matrix = np.identity(item_count)

    # Variable initialization
    alpha = 250
    beta = 10
    lam = 0.1
    mu = 50
    gamma = 0.1

    # Initialization random value(0..1) item_count*item_count matrices
    z1 = np.random.rand(item_count, item_count)
    z2 = np.random.rand(item_count, item_count)
    z3 = np.random.rand(item_count, item_count)
    z4 = np.random.rand(item_count, item_count)
    y1 = y2 = y3 = y4 = 0

    count = 0

    # Improving performance
    temp1 = (X.T * X) / mu
    tempx = 4 * identity_matrix
    tempy = (temp1 + tempx).I
    np.savetxt("/home/malintha/datasets/ml-100k/adikari1.txt", temp1, delimiter=",")

    while count < 15:

        # Obtaining W
        temp2 = temp1 + z1 + z2 + z3 + z4 + (y1 + y2 + y3 + y4) / mu
        W = tempy * temp2

        # Updating z1
        # Creating empty matrix n*n representing zero in element-wise operation
        temp3 = np.maximum(np.abs(W - y1 / mu) - alpha / mu, np.zeros(W.shape))
        temp4 = np.sign(W - y1 / mu)
        z1 = np.multiply(temp3, temp4)

        # Updating z2
        SVD = W - y2 / mu
        # calculating SVD using python Linear Algebra lib
        P, D, Q = np.linalg.svd(SVD, full_matrices=True)
        temp5 = np.maximum(np.diag(D) - beta / mu, 0)
        z2 = np.dot(np.dot(P, temp5), Q)

        # Updating z3
        S = np.zeros(shape=(item_count, item_count))
        for iterator1 in range(0, item_count):
            x_i = X[:, iterator1]
            for iterator2 in range(0, item_count):
                x_j = X[:, iterator2]
                # Sij = (xi.xj)/(||xi||2*||xj||2)
                if np.linalg.norm(x_i) * np.linalg.norm(x_j) == 0:
                    S[iterator1, iterator2] = 0
                else:
                    S[iterator1, iterator2] = np.dot(np.array(x_i).flatten(), np.array(x_j).flatten()) / np.linalg.norm(
                        x_i) * np.linalg.norm(x_j)

        # L = D - S
        L = np.diag(np.diag(S)) - S
        z3 = np.multiply(W - y3 / mu, np.matrix(lam / mu * L + identity_matrix).I)

        # Updating z4
        z4 = np.maximum(W - y4 / mu, 0)

        # Updating Lagrangian multipliers
        y1 += mu * (z1 - W)
        y2 += mu * (z2 - W)
        y3 += mu * (z3 - W)
        y4 += mu * (z4 - W)

        # Updating penalty parameter mu
        mu *= gamma

        count += 1
        evaluate(W)

    return W

###############################################################################
# Testing
y = np.matrix([[0 for i in range(number_of_movies)] for j in range(number_of_users)])

user_ratings = [[0] * number_of_movies for _ in range(number_of_users)]

###########################################################
# Data loading
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']


ratings_base = pand.read_csv('resources/data/ml-100k/u.data', sep='\t', names=r_cols,
                                 encoding='latin-1')

ratings_base.__delitem__('unix_timestamp')

ratings_test = DataFrame(columns=['user_id', 'movie_id', 'rating'])
user_count = 0
while user_count < number_of_users:
    selection = ((ratings_base.loc[ratings_base['user_id'] == user_count + 1]).sort_values(by='rating', ascending=0)).iloc[0]
    ratings_test.loc[user_count] = selection
    ratings_base.drop(selection.name, inplace=True)
    user_count += 1

for index, row in ratings_base.iterrows():
    user_ratings[row[0] - 1][row[1] - 1] = row[2]

weight_matrix = LapLRSFunc(np.asmatrix(user_ratings))