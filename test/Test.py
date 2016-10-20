#!/usr/bin/python

import numpy as np
from scipy import spatial

def LapLRSFunc(X):

    item_count = X.shape[1]

    # creating Identity matrix
    identity_matrix = np.identity(item_count)

    # Variable initialization
    alpha = 250
    beta = 10
    lam = 0.1
    mu = 50
    gamma = 1.5

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

    while count < 10:

        # Obtaining W
        print("Obtaining W...")
        temp2 = temp1 + z1 + z2 + z3 + z4 + (y1 + y2 + y3 + y4) / mu
        W = tempy * temp2

        # Updating z1
        # Creating empty matrix n*n representing zero in element-wise operation
        print("Updating Z1...")
        temp3 = np.maximum(np.abs(W -y1/mu) - alpha / mu, np.zeros(W.shape))
        temp4 = np.sign(W-y1/mu)
        z1 = np.multiply(temp3, temp4)

        # Updating z2
        print("Updating Z2...")
        SVD = W - y2 / mu
        # calculating SVD using python Linear Algebra lib
        P, D, Q = np.linalg.svd(SVD, full_matrices=True)
        temp5 = np.maximum(np.diag(D) - beta / mu, 0)
        z2 = np.dot(np.dot(P,temp5), Q)

        # Updating z3
        print("Updating Z3...")
        S = np.zeros(shape=(item_count,item_count))
        for iterator1 in range(0,item_count):
            x_i = X[:, iterator1]
            for iterator2 in range(0,item_count):
                x_j = X[:, iterator2]
                # Sij = (xi.xj)/(||xi||2*||xj||2)
                if(np.linalg.norm(x_i)* np.linalg.norm(x_j)==0):

                    S[iterator1, iterator2] = 0
                else:
                    S[iterator1, iterator2] = np.dot(np.array(x_i).flatten(), np.array(x_j).flatten())/np.linalg.norm(x_i)*np.linalg.norm(x_j)

        # L = D - S
        print("Calculating L")
        L = np.diag(np.diag(S)) - S
        z3 = np.multiply(W - y3 / mu, np.matrix(lam / mu * L + identity_matrix).I)

        # Updating z4
        print("Updating Z4...")
        z4 = np.maximum(W - y4 / mu, 0)

        # Updating Lagrangian multipliers
        y1 += mu * (z1 - W)
        y2 += mu * (z2 - W)
        y3 += mu * (z3 - W)
        y4 += mu * (z4 - W)

        # Updating penalty parameter mu
        mu = gamma * mu

        #print(W.shape)
        print(count)

        count += 1

    return W

###############################################################################
# Testing
y = np.matrix([[0 for i in range(1682)] for j in range(943)])

import pandas as pand
from pandas import *

user_ratings = [[0] * 1682 for _ in range(943)]

###########################################################
# Data loading
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pand.read_csv('resources/data/ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

ratings_base = pand.read_csv('resources/data/ml-100k/u2.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pand.read_csv('resources/data/ml-100k/u2.test', sep='\t', names=r_cols, encoding='latin-1')

###########################################################

ratings_base.__delitem__('unix_timestamp')
ratings.__delitem__('unix_timestamp')
ratings_test.__delitem__('unix_timestamp')
ratings_test.__delitem__('rating')

for index, row in ratings_base.iterrows():
    user_ratings[row[0]-1][row[1]-1] = row[2]
y= np.asmatrix(user_ratings)

#
# weight_matrix = LapLRSFunc(y)
#
# predicted = user_ratings * weight_matrix
# predicted = np.matrix.round(predicted)

#
 # Selecting top-N recommendations list
 # N=10
import bottleneck as bottleneck
# top_N_ratings = np.zeros(shape=(943,10))
# temp = np.squeeze(np.asarray(predicted))
# for (b,m), value in np.ndenumerate(temp):
#     top_N_ratings[b] = np.array(bottleneck.argpartsort(-temp[b], 10)[:10])
#
# Calculating

test_ratings = {k: g['movie_id'].tolist() for k,g in ratings_test.groupby('user_id')}
# for key,myvalue in test_ratings:
#     for movie_id in myvalue:
#         print
user_count= len(test_ratings)
hit_count=0
for key, myval in test_ratings:
    print(key)
    # for movie in myval:
    #     if movie in top_N_ratings[key]:
    #         hit_count = hit_count +1
    #         break
    # else:
    #     continue
    # break
# for (p,d), value in np.ndenumerate(top_N_ratings):
#     for val in top_N_ratings[p]:
#         if val in test_ratings[p]:
#             hit_count = hit_count +1
#

print(hit_count/user_ratings)


