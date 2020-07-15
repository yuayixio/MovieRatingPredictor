import pandas as pd
import unicodedata
import sys

from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def features(threshold_actors=20, ts_languages=10, year=True, runtime=True, imdbvotes=True, series=True, awards=True,
             genres=True, imdb_rating=True, roto_rating=True, pg_rating=True, threshold_newkeywords=200,
             threshold_plots=100, threshold_directors=5, metacritic=0):
    """

    :param threshold_actors:
    :param ts_languages:
    :param year:
    :param runtime:
    :param imdbVotes:
    :param series:
    :param awards:
    :param genres:
    :param imdb_rating:
    :param roto_rating:
    :param pg_rating:
    :param threshold_newkeywords:
    :param threshold_plots:
    :param threshold_directors:
    :param metacritic:
    :return:
    """
    # Preprocess omdb data
    omdb = pd.read_csv('../data/modeling/train/omdb_cleaned.csv')
    # Fill NaN Runtime
    omdb.loc[:, 'Runtime'] = omdb.loc[:, 'Runtime'].fillna(omdb.loc[:, 'Runtime'].median())
    # Fill NaN for imdbVotes
    omdb.loc[:, 'imdbVotes'] = omdb.loc[:, 'imdbVotes'].fillna(omdb.loc[:, 'imdbVotes'].median())
    # Combine awards in one column
    omdb['Awards'] = omdb['Oscars_won'] + omdb['Golden_globe_won'] + omdb['Oscars_nominated'] + omdb[
        'Golden_globe_nominated']
    omdb = omdb.drop(columns={'Oscars_won', 'Oscars_nominated', 'Golden_globe_won', 'Golden_globe_nominated'})
    omdb = omdb.rename(columns={"Rotten Tomatoes": "RottenTomatoes"})
    omdb['RottenTomatoes'] = omdb['RottenTomatoes'].where(~omdb['RottenTomatoes'].isna(), omdb['Metacritic'])

    # Replace Metacritic with RT Scroe if NaN
    omdb['Metacritic'].where(~omdb['Metacritic'].isna(), omdb['RottenTomatoes'])

    # Fill remaining with mean()
    omdb['RottenTomatoes'] = omdb['RottenTomatoes'].where(~omdb['RottenTomatoes'].isna(), omdb['RottenTomatoes'].mean())
    omdb['Metacritic'] = omdb['Metacritic'].where(~omdb['Metacritic'].isna(), omdb['Metacritic'].mean())

    # combine selected features
    # merge each selected feature with features dataframe
    features = omdb[['imdbID']]
    # store the selected feature combination
    names_features = []
    movies = pd.read_csv('../data/modeling/train/movies_id_updated.csv')
    mapping = movies[['id', 'imdbID']].rename(columns={'id': 'movieID'})
    # year
    if year:
        features = features.merge(omdb[['Year', 'imdbID']], on='imdbID', how='left')
        names_features.append('Year')
    # runtime
    if runtime:
        features = features.merge(omdb[['Runtime', 'imdbID']], on='imdbID', how='left')
        names_features.append('Runtime')
    # Number of Votes at imdb
    if imdbvotes:
        features = features.merge(omdb[['imdbVotes', 'imdbID']], on='imdbID', how='left')
        names_features.append('imdbVotes')
    # If Series or film
    if series:
        features = features.merge(omdb[['Series', 'imdbID']], on='imdbID', how='left')
        names_features.append('Series')
    # Number of awards won
    if awards:
        features = features.merge(omdb[['Awards', 'imdbID']], on='imdbID', how='left')
        names_features.append('Awards')
    # actors of the movies
    if threshold_actors != 0:
        actors = pd.read_csv('../data/raw/actors.csv', sep=',')
        # select only actors which appear in a number of movies larger than a certain threshold 
        actor_counts = pd.DataFrame(actors['actorID'].value_counts())
        actors_selected = actor_counts[actor_counts['actorID'] > threshold_actors]
        actors_selected = actors.set_index('actorID').loc[actors_selected.index].reset_index()
        # merge with imdbID, groupby imdbID and write the actors as one entry per movie
        actors_grouped = actors_selected.merge(mapping, on='movieID').groupby('imdbID')['actorID'].apply(
            list).reset_index(name='actors')
        # One-Hot Encoding with MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        actors_enc = pd.DataFrame(mlb.fit_transform(actors_grouped['actors']))
        actors_grouped = actors_grouped.join(actors_enc)
        actors_features = actors_grouped.drop(columns={'actors'})
        # enrich features
        features = features.merge(actors_features, on='imdbID', how='left')
        names_features.append('Actors: ' + str(threshold_actors))
    # genres
    if genres:
        genres = pd.read_csv('../data/raw/genres.csv', sep=',')
        genres_grouped = genres.merge(mapping, on='movieID').groupby('imdbID')['genre'].apply(list).reset_index(
            name='genres')
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
        lg_index = pd.DataFrame(lg.sum() > ts_languages)
        language = omdb[['0']].join(lg[lg_index[lg_index[0]].index]).rename(columns={'0': 'imdbID'})
        # enrich features
        features = features.merge(language, on='imdbID', how='left')
        names_features.append('Language: ' + str(ts_languages))
    # IMDB-Rating
    if imdb_rating:
        imdb_ratings = omdb[['0', 'imdbRating']].rename(columns={'0': 'imdbID'})
        features = features.merge(imdb_ratings, on='imdbID', how='left')
        names_features.append('imdb_ratings')
    # Rotten-Tomatoes Rating
    if roto_rating:
        roto_ratings = omdb[['0', 'RottenTomatoes']].rename(columns={'0': 'imdbID'})
        features = features.merge(roto_ratings, on='imdbID', how='left')
        names_features.append('RottenTomatoes_Rating')
    # Metacritic Rating
    if metacritic != 0:
        metacritic_ratings = omdb[['0', 'Metacritic']].rename(columns={'0': 'imdbID'})
        features = features.merge(metacritic_ratings, on='imdbID', how='left')
        names_features.append('Metacritic_Rating')
    # Parental Guidance Indication
    if pg_rating:
        pg_ratings = omdb[['0', 'PG_Rating']].rename(columns={'0': 'imdbID'})
        features = features.merge(pg_ratings, on='imdbID', how='left')
        names_features.append('PG_Rating')
    # Keywords from IMDB
    if threshold_newkeywords != 0:
        newkeywords = pd.read_csv('../ContentbasedFiltering/keywordDict.csv', header=None, sep=';')
        newkeywords = newkeywords.dropna()
        newkeywords[1] = newkeywords[1].apply(lambda x: x[1:-1])
        newkeywords[1] = newkeywords[1].apply(lambda x: x.split(','))
        newkeywords = newkeywords.explode(1)
        newkeywords_counts = pd.DataFrame(newkeywords[1].value_counts())
        newkeywords_selected = newkeywords_counts[newkeywords_counts[1] > threshold_newkeywords]
        newkeywords_selected = newkeywords.set_index(1).loc[newkeywords_selected.index].reset_index()
        newkeywords_selected = newkeywords_selected.rename(columns={0: 'imdbID'})
        newkeywords_grouped = newkeywords_selected.groupby('imdbID')[1].apply(list).reset_index(name='newkeywords')
        mlb = MultiLabelBinarizer()
        newkeywords_enc = pd.DataFrame(mlb.fit_transform(newkeywords_grouped['newkeywords']))
        newkeywords_grouped = newkeywords_grouped.join(newkeywords_enc).drop(columns={'newkeywords'})
        features = features.merge(newkeywords_grouped, on='imdbID', how='left')
        names_features.append('NewKeywords: ' + str(threshold_newkeywords))
    # Keywords extracted from plot
    if threshold_plots != 0:
        plots = pd.read_csv('../data/preprocessed/plot.csv')
        plots = plots.dropna()
        punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                                    if unicodedata.category(chr(i)).startswith('P'))
        plots['Plot'] = [string.translate(punctuation) for string in plots['Plot']]
        plots['Plot'] = plots['Plot'].apply(word_tokenize)
        plots['Plot'] = plots['Plot'].apply(lambda x: [item.lower() for item in x])
        stop_words = stopwords.words('english') + ['find', 'one', 'two', 'three', 'four', 'set', 'film', 'come', 'get',
                                                   'take', 'must', 'film', 'make', 'go', 'high', 'former', 'look',
                                                   'movie', 'make', 'go', 'high', 'us', 'use', 'whose', 'stop', 'sent',
                                                   'series', 'another', 'arrive', 'ii', 'bring', 'see', 'big', 'keep',
                                                   'cause', 'because', 'he', 'leave']
        plots['Plot'] = plots['Plot'].apply(lambda x: [item for item in x if item not in stop_words])
        porter = PorterStemmer()
        plots['Plot'] = plots['Plot'].apply(lambda x: [porter.stem(word) for word in x])
        plots = plots.explode('Plot')
        plots_counts = pd.DataFrame(plots['Plot'].value_counts())
        plots_selected = plots_counts[plots_counts['Plot'] > threshold_plots]
        plots_selected = plots.set_index('Plot').loc[plots_selected.index].reset_index()
        plots_grouped = plots_selected.groupby('imdbID')['Plot'].apply(list).reset_index(name='plots')
        mlb = MultiLabelBinarizer()
        plots_enc = pd.DataFrame(mlb.fit_transform(plots_grouped['plots']))
        plots_grouped = plots_grouped.join(plots_enc).drop(columns={'plots'})
        features = features.merge(plots_grouped, on='imdbID', how='left')
        names_features.append('Plots: ' + str(threshold_plots))
    # Directors
    if threshold_directors != 0:
        directors = pd.read_csv('../data/raw/directors.csv', sep=',')
        director_counts = pd.DataFrame(directors['directorID'].value_counts())
        directors_selected = director_counts[director_counts['directorID'] > threshold_directors]
        directors_selected = directors.set_index('directorID').loc[directors_selected.index].reset_index()
        # merge with imdbID, groupby imdbID and write the x most prominent directors as one entry per movie
        directors_grouped = directors_selected.merge(mapping, on='movieID').groupby('imdbID')['directorID'].apply(
            list).reset_index(name='directors')
        mlb = MultiLabelBinarizer()
        directors_enc = pd.DataFrame(mlb.fit_transform(directors_grouped['directors']))
        directors_grouped = directors_grouped.join(directors_enc).drop(columns={'directors'})
        features = features.merge(directors_grouped, on='imdbID', how='left')
        names_features.append('Directors: ' + str(threshold_directors))

    # fill nan values
    features = features.fillna(0)

    # Standardize the features
    scaler = preprocessing.StandardScaler()
    features = scaler.fit_transform(features.drop(columns={'imdbID'}))

    return features, names_features
