import pandas as pd
import numpy as np
import string
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
import time
kwargs = dict(random_state=42)

def features(threshold_actors=20, ts_languages=10, year=True, runtime=True, imdbVotes=True, series=True, awards=True, genres=True, rating=True, rated=True):
    # Preprocess omdb data
    omdb = pd.read_csv('../../data/preprocessed/omdb_cleaned.csv')
    # Fill NaN Runtime
    omdb.loc[:, 'Runtime'] = omdb.loc[:, 'Runtime'].fillna(omdb.loc[:, 'Runtime'].median())
    # Fill NaN for imdbVotes
    omdb.loc[:, 'imdbVotes'] = omdb.loc[:, 'imdbVotes'].fillna(omdb.loc[:, 'imdbVotes'].median())
    #Combine awards in one column
    omdb['Awards'] = omdb['Oscars_won'] + omdb['Golden_globe_won'] + omdb['Oscars_nominated'] + omdb['Golden_globe_nominated']
    omdb = omdb.drop(columns={'Oscars_won', 'Oscars_nominated', 'Golden_globe_won', 'Golden_globe_nominated'})
    
    # combine selected features
    # merge each selected feature with features dataframe
    features = omdb[['imdbID']]
    # store the selected feature combination
    names_features = []
    movies = pd.read_csv('../../data/preprocessed/movies_id_updated.csv')
    mapping = movies[['id', 'imdbID']].rename(columns={'id':'movieID'})
    # year
    if year == True:
        features = features.merge(omdb[['Year', 'imdbID']], on='imdbID', how='left')
        names_features.append('Year')
    # runtime
    if runtime == True:
        features = features.merge(omdb[['Runtime', 'imdbID']], on='imdbID', how='left')
        names_features.append('Runtime')
    # Number of Votes at imdb
    if imdbVotes == True:
        features = features.merge(omdb[['imdbVotes', 'imdbID']], on='imdbID', how='left')
        names_features.append('imdbVotes')
    # If Series or film
    if series == True:
        features = features.merge(omdb[['Series', 'imdbID']], on='imdbID', how='left')
        names_features.append('Series')
    # Number of awards won
    if awards == True:
        features = features.merge(omdb[['Awards', 'imdbID']], on='imdbID', how='left')
        names_features.append('Awards')
    # actors of the movies
    if threshold_actors != 0:
        actors = pd.read_csv('../../data/raw/actors.csv', sep=',')
        # select only actors which appear in a number of movies larger than a certain threshold 
        actor_counts = pd.DataFrame(actors['actorID'].value_counts())
        actors_selected = actor_counts[actor_counts['actorID']>threshold_actors]
        actors_selected = actors.set_index('actorID').loc[actors_selected.index].reset_index()
        # merge with imdbID, groupby imdbID and write the actors as one entry per movie
        actors_grouped = actors_selected.merge(mapping, on='movieID').groupby('imdbID')['actorID'].apply(list).reset_index(name='actors')
        # One-Hot Encoding with MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        actors_enc = pd.DataFrame(mlb.fit_transform(actors_grouped['actors']))
        actors_grouped = actors_grouped.join(actors_enc)
        actors_features = actors_grouped.drop(columns={'actors'})
        # enrich features
        features = features.merge(actors_features, on='imdbID', how='left')
        names_features.append('Actors: '+str(threshold_actors))
    # genres
    if genres == True: 
        genres = pd.read_csv('../../data/raw/genres.csv', sep=',')
        genres_grouped = genres.merge(mapping, on='movieID').groupby('imdbID')['genre'].apply(list).reset_index(name='genres')
        # One-Hot Encoding with MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(genres_grouped['genres'])
        genres_grouped = genres_grouped.join(pd.DataFrame(genres_encoded))
        genres_grouped = genres_grouped.sort_values('imdbID').drop(columns={'genres'})
        # enrich features
        features = features.merge(genres_grouped, on='imdbID', how='left')
        names_features.append('Genres')
    # language of the movies
    if ts_languages != 0: 
        mlb = MultiLabelBinarizer()
        lg = pd.DataFrame(mlb.fit_transform(omdb['Language']))
        # select languages which appear in more than movies than the threshold
        lg_index = pd.DataFrame(lg.sum()>ts_languages)
        language = omdb[['0']].join(lg[lg_index[lg_index[0]].index]).rename(columns={'0':'imdbID'})
        # enrich features
        features = features.merge(language, on='imdbID', how='left')
        names_features.append('Language: '+str(ts_languages))
    
    if rating==True:
        imdb_ratings = omdb[['0', 'imdbRating']].rename(columns={'0':'imdbID'})
        features = features.merge(imdb_ratings, on='imdbID', how='left')
        names_features.append('imdb_ratings')
    if rated==True:
        PG_Rating = omdb[['0', 'PG_Rating']].rename(columns={'0':'imdbID'})
        features = features.merge(imdb_ratings, on='imdbID', how='left')
        names_features.append('PG_Rating')
    
    # fill nan values
    features = features.fillna(0)
    
    # Standardize the features
    scaler = preprocessing.StandardScaler()
    features = scaler.fit_transform(features.drop(columns = {'imdbID'}))
    
    return features, names_features


def calculate_KNN(features, metric='minkowski'):
    # create NearesNeighbor model with the minkowski distance
    knn = NearestNeighbors(metric=metric, p=2, algorithm='brute', n_jobs=-1)
    
    # fit to all the features
    knn.fit(features)
    
    # create two array for distance and index to store
    distances = np.empty((features.shape[0],features.shape[0]))
    indices = np.empty((features.shape[0],features.shape[0]))

    # iterate over all imdbIDs
    print('Calculating Nearest Neighbors and distances for all movies')
    for i in range(0, features.shape[0]):
        # get all neighbors and the distances for each imdbID i
        distances[i], indices[i] = knn.kneighbors([features[i]], n_neighbors=features.shape[0])
        
        if i%1000==0:
            print(i)
    print('All Nearest Neighbors and distances calculated')
    # convert in dataframe
    distances_df = pd.DataFrame(distances)
    indices_df = pd.DataFrame(indices)
    return distances_df, indices_df

# Use optimal k
def adjust_k(r_size):
    adjusted_k = 10
    if r_size > 40 and r_size  < 100:
        adjusted_k = 15
    elif r_size  > 100 and r_size < 500:
        adjusted_k = 20
    elif r_size  > 500 and r_size < 1500:
        adjusted_k = 25
    elif r_size  > 1500:
        adjusted_k = 30
        
    return adjusted_k

#Alles in eine Funktion
def predict_rating(index, movie_row, movie_row_distance, indices, n_neighbors, rated_id, imdbID, userID, mean=True):
    #zeitu = time.perf_counter()
    
    #Speichere Reihenfolge der Indizes der Nachbarn des Films und tausche erste Reihe mit Spaltenbezeichnungen
    movie_row = movie_row.reset_index(drop=True)
    movie_row.columns = movie_row.loc[0]
    movie_row.drop(0, inplace=True)
    movie_row.loc[1,:] = range(0,n_neighbors)
    
    #Speichere alle Filme, die der User bewertet hat und bestimme index davon
    rated_df = rated_id['index']
    rated_df = pd.DataFrame(rated_df)
    
    # Bestimme die Position der bewerteten Filme innerhalb der Nachbarschaft
    position = movie_row[list(rated_df['index'].astype(int))]
    position = position.transpose()
    position.columns = ['position']
    position ['index'] = position.index
    position = position.reset_index(drop = True)
    
    #Bestimme Distanz der Filme vom Film, die der User bewertet hat
    user_distances = movie_row_distance[list(position['position'].astype(int))].transpose()
    user_distances.columns = ['distance']
    user_distances =user_distances.reset_index(drop = True)
    
    #Füge alle Infos zusammen und sortiere von nächstem zum weitest entfernten Film
    neighbors = pd.concat([position, user_distances], axis=1, join = "inner")
    neighbors = neighbors.sort_values("distance")
        
    #Füge die Ratings des Users zu diesen Filmen noch hinzu
    neighbors = neighbors.merge(rated_id[["rating","index"]], on="index", how = "outer")
    neighbors = neighbors.iloc[1:]
    
    #Berechne die Prognose: 
    if mean==True:
        # Mittelwert der k-nächsten Nachbarn
        k = adjust_k(neighbors.shape[0])
        pred = neighbors['rating'].iloc[:k].mean()
    else:
        # Gewichtung der Bewertung der Nachbarn je nach Distanz
        pred = sum(neighbors['rating']*((1/neighbors['distance'])**10)/(sum((1/neighbors['distance'])**10)))
    
    return pred