import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

def read_files():
    # get directory name of running file, get relative path for ratings and movies
    dirname = os.path.dirname(__file__)
    ratings = os.path.join(dirname, '../Data/ml-latest/ratings.csv')
    movies = os.path.join(dirname, '../Data/ml-latest/movies.csv')

    # get p
    df_ratings=pd.read_csv(ratings, usecols=['movieId','userId','rating'],
        dtype={
            'movieId':'int32',
            'userId':'int32',
            'rating':'float32',
            }
        )

    df_movies=pd.read_csv(movies, usecols=['movieId','genres','title'],
        dtype={
            'movieId':'int32',
            'genres':'str',
            'title':'str',
            }
        )

    return df_ratings, df_movies

def create_sparse_matrix(df_ratings, df_movies):
    """
    Creates a sparse matrix with movie ratings as values, movie IDs as rows, and user IDs as columns.

    Args:
    ratings_df (pandas.DataFrame): DataFrame containing movie ratings and IDs.
    movies_df (pandas.DataFrame): DataFrame containing movie IDs and titles.

    Returns:
    csr_matrix: Sparse matrix with movie ratings.
    """

    # categorical variable for movieId column in ratings and userId in users
    movies = pd.Categorical(df_ratings['movieId'], categories=df_movies['movieId'])
    users = pd.Categorical(df_ratings['userId'])

    # sparse matrix with ratings as values, movieId as rows, and userId as columns
    return csr_matrix((df_ratings['rating'], (movies.codes, users.codes)))

def create_model(movie_user_ratings_matrix):
    """
    Creates a nearest neighbors model with cosine similarity as distance metric and brute force as algorithm.

    Args:
        movie_user_ratings_matrix (scipy.sparse.csr_matrix): Sparse matrix.

    Returns:
        sklearn.neighbors._unsupervised.NearestNeighbors: Nearest neighbors model.
    """

    model_KNN = NearestNeighbors(metric='cosine', algorithm='brute')
    
    model_KNN.fit(movie_user_ratings_matrix) # fits model to sparse matrix

    return model_KNN

def recommender(movie_name, df_movies, model_KNN, data, num_recommendations = 5):
    """
    Returns a list of recommended movies based on the input movie name.

    Args:
        movie_name (str): Name of the movie for which recommendations are sought.
        movies_df (pandas.DataFrame): Dataframe of movies with columns movieId and title.
        model_knn (sklearn.neighbors._unsupervised.NearestNeighbors): Nearest neighbors model.
        ratings_matrix (scipy.sparse.csr_matrix): Sparse matrix of ratings with movieId as rows and userId as columns.
        num_recommendations (int): Number of recommended movies to return.
    
    Returns: Prints num_recommendations movies.
    """

    # find index of movie in dataframe that matches input movie name
    selected_movie_index = process.extractOne(movie_name, df_movies['title'])[2]
    # find nearest neighbors of selected movie
    distances, indices = model_KNN.kneighbors(data[selected_movie_index], n_neighbors = num_recommendations + 1)

    # remove first index, which is selected movie itself
    indices = indices.flatten()[1:]
    
    # print selected movie title
    print(f"Recommendations for {df_movies.loc[selected_movie_index]['title']}:\n")

    # print recommended movies in order from closest to farthest
    for a, i in enumerate(indices):
        print(f"{a + 1}. {df_movies.loc[i]['title']}\n")

def input_movie():
    movie_name = input("Search movie: ")
    return movie_name

def cls():
    # clear console
    os.system('cls' if os.name=='nt' else 'clear')

if __name__ == '__main__':
    print('Reading files...')
    df_ratings, df_movies = read_files()

    print('Creating model instance...')
    matrix = create_sparse_matrix(df_ratings, df_movies)
    model_KNN = create_model(matrix)

    cls()

    while True:
        movie_name = input_movie()
        recommender(movie_name, df_movies, model_KNN, matrix, num_recommendations=5)