import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

from knn_preprocessing import knn_preprocessing

kwargs = dict(random_state=42)
## function that predicts the rating of a movie from its imdbID and its nearest neighbors

def compute_rating(neighbors, distances, mean = False):
    
    if mean == True:
        pred = neighbors.mean()
    else:
        #scaling ratings based on distance
        pred = sum(neighbors* (1+(1-distances[0]/distances[0].mean()))) / neighbors.shape[0]
    
    return float(pred)

# First Approach for easy k tuning - use method later to implement in depth tuning of k

def adjust_k(ratings):
    adjusted_k = 10
    r_size = len(ratings)
    #adjusted_k = int(math.sqrt(r_size))
    if r_size > 40 and r_size  < 100:
        adjusted_k = 15
    elif r_size  > 100 and r_size < 500:
        adjusted_k = 20
    elif r_size  > 500 and r_size < 1500:
        adjusted_k = 25
    elif r_size  > 1500:
        adjusted_k = 30
        print(r_size) 
    return adjusted_k

def predict_movie_rating(imdbID, userID, user_data, mean=False, knn_metric='cosine', set_k=False, k_neighbors=15):
   
    # Select all ratings given by User #userID
    ratings = user_data.loc[user_data['user_id'] == userID]
    
    #If no explicit number of neighbors is passed -> use variable neighbors function
    if set_k:
        k_neighbors = k_neighbors
    else:    
        k_neighbors = adjust_k(ratings)
  
    # Get real rating -> remove this in the end -> currently done for validation
    real_ratings = ratings.loc[(ratings['imdbID'] == imdbID)]
    real_idx = ratings.loc[(ratings['imdbID'] == imdbID)].index
    
    #remove real rating
    ratings = ratings[ratings['imdbID'] != imdbID] 

    #Scaling features -> maybe do outside function in future
    scaler = preprocessing.StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(ratings.drop(columns = {'imdbID','user_id', 'rating'}))).merge(pd.DataFrame(ratings.index), left_index=True, right_index=True, how='left')
          
    if (ratings.to_numpy().size>0):   
        # Set algorithm and params
        if knn_metric == 'minkowski':
            knn = NearestNeighbors(metric='minkowski',p=2 , algorithm='brute', n_neighbors=k_neighbors, n_jobs=-1)
        else:    
            knn = NearestNeighbors(metric=knn_metric , algorithm='brute', n_neighbors=k_neighbors, n_jobs=-1)
    
        knn.fit(csr_matrix(features.iloc[:,0:(user_data.shape[1]-3)]))
        
        input_data = user_data.iloc[real_idx]
        inputs = scaler.transform(input_data.drop(columns = {'imdbID','user_id', 'rating'}))
         
        #Prediction -> get x nearest neighbors of imdbID
        distances , indices = knn.kneighbors(inputs, n_neighbors=k_neighbors)
        
        # Zieht indices und ratings der neighbors
        neighbor_ratings = user_data['rating'].loc[features['0_y'].loc[indices[0]]]
      
        # compute rating of movie(imbdID) based on the rating of the 20 nearest neighbors
        #mean = True gibt nur mittelwert der nachbarn
        
        pred = compute_rating(neighbor_ratings, distances, mean)
    
        # return rating prediction and real rating
        return pred , real_ratings['rating'].values[0]
        
    else:
         return "User has not rated other movies. Check input"
    