# your predictions go here
import pandas as pd


def predict_ratings():
    """
    prediction function for assessment submission
    :return: returns test_data and test_data_with_rating
    """

    test_data = pd.read_csv("../data/modeling/test/test.csv")
    test_data_with_rating = pd.DataFrame(columns=["userID","movieID","rating"])
    
    # loop through test data
    for index, row in test_data.iterrows():
        
        user_id = test_data['user_id'][index]
        movie_id = test_data['user_id'][index]

        # Insert Prediction function and get results as return
    
        # get userID, movieID and rating -> wanted output format
        rating = RATINGPREDICTIONFUNCTION(user_id, movie_id)
    
        # make use of helper function to put prediction into the right shape
        test_data_with_rating = putRatingsIntoFormat(user_id, movie_id, rating, test_data_with_rating)
    
    return test_data, test_data_with_rating


############## Function to get prediction into the right shape ##########################

def putRatingsIntoFormat (userID, movieID, rating, output_df = pd.DataFrame(columns=["userID","movieID","rating"])):
    
    ratingList = {'userID':userID, "movieID":movieID, "rating":rating}
    output_df = output_df.append(ratingList, ignore_index=True)
    
    return output_df