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
    # create a categorical variable for movieId column in ratings and userId in ratings
    movies = pd.Categorical(df_ratings['movieId'], categories=df_movies['movieId'])
    users = pd.Categorical(df_ratings['userId'])

    # create a sparse matrix with ratings as values, movieId as rows, and userId as columns
    movie_user_ratings_matrix = csr_matrix((df_ratings['rating'], (movies.codes, users.codes)))

    return movie_user_ratings_matrix

def create_model(movie_user_ratings_matrix):
    # create a nearest neighbors model with cosine similarity as distance metric and brute force as algorithm
    model_KNN = NearestNeighbors(metric='cosine', algorithm='brute')
    
    # fit model to sparse matrix
    model_KNN.fit(movie_user_ratings_matrix)

    return model_KNN

def recommender(movie_name, df_movies, model_KNN, data, num_recommendations = 5):
    # find index of movie in dataframe that matches input movie name
    idx = process.extractOne(movie_name, df_movies['title'])[2]

    # find nearest neighbors of selected movie
    distances, indices = model_KNN.kneighbors(data[idx], n_neighbors = num_recommendations + 1)

    # remove first index, which is selected movie itself
    indices = indices.flatten()[1:]
    
    # print selected movie title
    print(f"Recommendations for {df_movies.loc[idx]['title']}:\n")

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