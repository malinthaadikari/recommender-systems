import pandas as pand
from pandas import *
from LapLRS import LapLRSFunc

number_of_users = 943
number_of_movies = 1682
selected_recommendations = 10

def data_pre_processor(data_file_location, number_of_folds):
    # Data loading into pandas dataframe and formatting
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    ratings_base = pand.read_csv(data_file_location, sep='\t', names=r_cols,
                                 encoding='latin-1')

    ratings_base.__delitem__('unix_timestamp')

    # Splitting data into K folds
    k_folds = np.array_split(ratings_base, number_of_folds)

    # Repeat this for each fold
    for fold in k_folds:
        # Create new Dataframe to store testset
        ratings_test_set = DataFrame(columns=['user_id', 'movie_id', 'rating'])

        # Collecting ratings for each user
        user_count = 0
        while user_count < number_of_users:
            # if the current user is not in the fold we skip adding new row to the testset
            if len(fold.loc[fold['user_id'] == user_count + 1]) != 0:
                # We select all ratings of the user then we sort by rating and finally choose the top rated movie
                selection = ((fold.loc[fold['user_id'] == user_count + 1]).sort_values(by='rating', ascending=0)).iloc[
                    0]

                # Put thar movie into testset
                ratings_test_set.loc[user_count] = selection

                # Remove that movie from the training set (Leave one out)
                fold.drop(selection.name, inplace=True)
            user_count += 1
        # End of the above loop we have testset which contains one top rated movie for each user, these are hidden items

        # Creating training set, size is number of users in current fold
        ratings_training_set = [[0] * number_of_movies for _ in range(len(unique(fold['user_id'])))]

        # Populating the training set
        # row[0]-1 is the user id
        # row[1]-1 is the movie_id
        # row[2] is the rating value
        for index, row in fold.iterrows():
            ratings_training_set[row[0] - 1][row[1] - 1] = row[2]

        return ratings_training_set, ratings_test_set
