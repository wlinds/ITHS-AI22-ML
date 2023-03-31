import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Data/spam.csv", encoding="latin-1").dropna(axis=1)
df = pd.get_dummies(df, columns=['v1'], drop_first=True)

X, y = df['v2'], df['v1_spam']

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf_vectorizer.fit_transform(X)

new_text = "Hello, I am a string, I am a text string to analyze."

# Transform the new text using the trained TfidfVectorizer
new_text_tfidf = tfidf_vectorizer.transform([new_text])

print(new_text_tfidf.toarray()) # vector representation of the new text
print(new_text_tfidf)