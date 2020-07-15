import pandas as pd

from train import make_prediction

def predict_ratings():
    """
    prediction function for assessment submission
    :return: returns test_data and test_data_with_rating
    """
    # read in test data
    test_data = pd.read_csv("../data/modeling/test/test.csv")

    # read in stored movies to map movieID to unique imdbID
    movies = pd.read_csv("../data/modeling/train/movies_id_updated.csv", sep=',')
    mapping = movies[['id', 'imdbID']].rename(columns={'id': 'movieID'})
    test_data_imdb = test_data.merge(mapping, how='left', on='movieID')
    
    # make prediction
    test_data_with_rating = make_prediction(test_data_imdb)

    # put prediction into the right shape
    test_data_with_rating = test_data_with_rating[['user_id', 'movieID', 'rating']]

    return test_data, test_data_with_rating