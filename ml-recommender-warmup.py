import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import time

movies='Data/ml-latest/movies.csv'
ratings='Data/ml-latest/ratings.csv'

t1 = time.time()

dfp_movies=pl.read_csv(movies, infer_schema_length=0, columns=['movieId','title'])
dfp_ratings=pl.read_csv(ratings, columns=['userId','movieId','rating'],
    dtypes={
        'userId': pl.Int32,
        'movieId': pl.Int32,
        'rating': pl.Float32
        }
    )

t2 = time.time()
print(f'Took {t2-t1} seconds, {type(dfp_movies)}.')

t1 = time.time()
movies_users = dfp_ratings.pivot(index='movieId', columns='userId', values='rating')

t2 = time.time()
print(f'Took {t2-t1} seconds.')