import pandas as pd
import numpy as np

from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans, SVD

from sklearn.neighbors import NearestNeighbors

from src import preprocessing


def make_prediction(test_data_imdb):
    train_data = pd.read_csv('../data/modeling/train/ratings_clean_std_0.csv', sep=',').drop(columns={'Unnamed: 0'})
    omdb = pd.read_csv('../data/modeling/train/omdb_cleaned.csv')

    # build a reader, define the rating scale (minimum and maximum value)
    reader = Reader(rating_scale=(0.5, 5))
    # convert data to surprise format
    train_surprise = Dataset.load_from_df(train_data, reader).build_full_trainset()

    # Collaborative Filtering Models
    knn_collaborative = KNNWithMeans(k=115, min_k=5, sim_options={'name': 'msd', 'user_based': False})
    knn_collaborative.fit(train_surprise)
    svd = SVD(lr_all=0.01, reg_all=0.05, n_epochs=23)
    svd.fit(train_surprise)
    preds = [[knn_collaborative.predict(test[1], test[3]).est for test in test_data_imdb.itertuples()],
             [svd.predict(test[1], test[3]).est for test in test_data_imdb.itertuples()]]

    # Content-Based Models
    # define features for content-based models
    params_features = {'threshold_actors': 0, 'ts_languages': 0, 'year': True, 'runtime': True, 'imdbvotes': True,
                       'series': False, 'awards': False, 'genres': True, 'imdb_rating': True, 'roto_rating': True,
                       'pg_rating': True, 'threshold_newkeywords': 0, 'threshold_plots': 0, 'threshold_directors': 200}
    # load features
    features, names = preprocessing.features(**params_features)

    # add imdbID and set as index
    features = omdb[['imdbID']].join(pd.DataFrame(features)).set_index('imdbID')

    # predict ratings
    pred_content = []
    no_of_ratings = []
    for row in test_data_imdb.itertuples():
        # select user and movie
        imdbID = row.imdbID
        userID = row.user_id
        # select ratings of the user
        ratings_user = train_data.loc[train_data['user_id'] == userID]
        ratings_user.reset_index(inplace=True, drop=True)

        # select features of corresponding movies and convert to array
        features_user = np.array(features.loc[ratings_user['imdbID']])
        features_movie = np.array(features.loc[imdbID])

        # compute predictions
        if imdbID == 'tt0720339':
            pred_content.append(svd.predict(userID, imdbID).est)
        else:
            pred_content.append(predict_movie_rating(ratings_user, features_user, features_movie))
        # store the number of predictions of a user:
        no_of_ratings.append(ratings_user.shape[0])

    # predictions of the models
    predictions = weighted_prediction(preds[0], preds[1], pred_content, no_of_ratings)
    test_data_with_rating = test_data_imdb.join(predictions)

    return test_data_with_rating[['user_id', 'movieID', 'rating']]


def weighted_prediction(knn_colab, svd, content_based, no_of_ratings):
    df = pd.DataFrame()
    df['knn_colab'] = knn_colab
    df['svd'] = svd
    df['content_based'] = content_based
    df['no_of_ratings'] = no_of_ratings

    df['rating'] = df.apply(f, axis=1)
    return df[['rating']]


# calculate weighted prediction with optimal weights
def f(x):
    if x['no_of_ratings'] <= 350:
        return .0 * x['knn_colab'] + .88 * x['svd'] + .12 * x['content_based']
    elif 350 < x['no_of_ratings'] <= 650:
        return .0 * x['knn_colab'] + .88 * x['svd'] + .12 * x['content_based']
    elif 650 < x['no_of_ratings'] <= 1050:
        return .22 * x['knn_colab'] + .60 * x['svd'] + .19 * x['content_based']
    elif 1050 < x['no_of_ratings']:
        return .21 * x['knn_colab'] + .60 * x['svd'] + .19 * x['content_based']


def predict_movie_rating(ratings_user, features_user, features_movie):
    # If no explicit number of neighbors is passed -> use variable neighbors function
    k_neighbors = adjust_k(ratings_user)

    # Set algorithm and params
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k_neighbors, n_jobs=-1)

    ratings_user.reset_index(inplace=True, drop=False)

    # train algorithm
    knn.fit(features_user)

    # generate input data
    input_data = features_movie.reshape(1, -1)

    # Prediction -> get x nearest neighbors of imdbID
    distances, indices = knn.kneighbors(input_data, n_neighbors=k_neighbors)

    # Zieht indices und ratings der neighbors
    neighbor_ratings = ratings_user['rating'].loc[indices[0]]

    # Calculate rating
    pred = compute_rating(neighbor_ratings, distances)

    return pred


# function that computes a rating based on the neighbors
def compute_rating(neighbors, distances):
    # Gewichtung der Bewertung der Nachbarn je nach Distanz
    pred = sum(neighbors * ((1 / (distances[0] + 0.000001) ** 1) / (sum((1 / (distances[0] + 0.000001) ** 1)))))

    return float(pred)


# Use optimal k based on # rated movies
def adjust_k(ratings_k):
    adjusted_k = 10
    r_size = len(ratings_k)

    if 40 < r_size < 100:
        adjusted_k = 15
    elif 100 < r_size < 500:
        adjusted_k = 20
    elif 500 < r_size < 1500:
        adjusted_k = 25
    elif r_size > 1500:
        adjusted_k = 30

    return adjusted_k
