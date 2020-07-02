### kNN Preprocessing
import pandas as pd


def knn_preprocessing(omdb_columns):
    movies = pd.read_csv('../../data/preprocessed/movies_id_updated.csv')
    omdb = pd.read_csv('../../data/preprocessed/omdb_cleaned.csv')
    ratings = pd.read_csv('../../data/preprocessed/ratings_clean_std_0.csv')

    # Select relevant columns
    omdb = omdb[omdb_columns]

    # Add movies which could not be retrieved by omdb
    imdb_ids = pd.DataFrame(movies['imdbID'].unique()).rename(columns={0: 'imdbID'})
    omdb = imdb_ids.merge(omdb, how='left', on='imdbID')

    # Dropping the 10 almost empty movies
    indices = omdb[omdb['imdbRating'].isna()]['imdbID'].index
    for i in indices:
        omdb = omdb.drop([i], axis=0, )

    # Replace for Series, PG and awards NaN with 0 and handle accordingly
    for i in range(9, 14):
        omdb.iloc[:, i] = omdb.iloc[:, i].fillna(0)

    # Fill NaN Runtime
    omdb.loc[:, 'Runtime'] = omdb.loc[:, 'Runtime'].fillna(omdb.loc[:, 'Runtime'].median())
    # Fill NaN for imdbVotes
    omdb.loc[:, 'imdbVotes'] = omdb.loc[:, 'imdbVotes'].fillna(omdb.loc[:, 'imdbVotes'].median())
    # for i in range (4,16):
    # omdb.iloc[:,i] = omdb.iloc[:,i].fillna(omdb.iloc[:,i].median())

    omdb = omdb.rename(columns={"Rotten Tomatoes": "RottenTomatoes"})

    # Replace RT Score with Metacritic if NaN
    # where Replace values where the condition is False.
    omdb['RottenTomatoes'] = omdb['RottenTomatoes'].where(~omdb['RottenTomatoes'].isna(), omdb['Metacritic'])

    # Replace Metacritic with RT Scroe if NaN
    omdb['Metacritic'].where(~omdb['Metacritic'].isna(), omdb['RottenTomatoes'])

    # Fill remaining with mean()
    omdb['RottenTomatoes'] = omdb['RottenTomatoes'].where(~omdb['RottenTomatoes'].isna(), omdb['RottenTomatoes'].mean())
    omdb['Metacritic'] = omdb['Metacritic'].where(~omdb['Metacritic'].isna(), omdb['Metacritic'].mean())

    merged_data = ratings.merge(omdb, how='left', on='imdbID')
    merged_data = merged_data.drop(columns={'Unnamed: 0', 'Language'})

    # Comment Max: No NaN rows anymore - except for language for the missing movies
    merged_data.isna().sum()

    # convert imdbID from string to float
    merged_data['imdbID'] = merged_data['imdbID'].str.replace(r'tt', '')
    merged_data['imdbID'] = merged_data['imdbID'].astype(float)

    # Jetzt einfach ma5 mean() eingefüllt
    for i in range(3, 15):
        merged_data.iloc[:, i] = merged_data.iloc[:, i].fillna(merged_data.iloc[:, i].median())

    return merged_data
