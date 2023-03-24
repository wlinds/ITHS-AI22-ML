#Skapa ett nytt skript: **production_model.py**, ladda in **test_samples.csv** och din modell. Använd
#```joblib.load()``` för att ladda in en .pkl-fil. Gör prediction på de 100 datapunkterna och exportera en fil "prediction.csv" som ska innehålla kolumnerna med ifyllda värden:
#
#- probability class 0
#- probability class 1
#- prediction

import pandas as pd
import joblib
import os

#print(os.getcwd())

test_data = pd.read_csv('Labb/test_samples.csv')
test_data = test_data.drop('cardio', axis=1)

model = joblib.load('Labb/trained_model.pkl')

proba = model.predict_proba(test_data)
predictions = model.predict(test_data)

results = pd.DataFrame({'probability class 0': proba[:,0], 'probability class 1': proba[:,1], 'prediction': predictions})

results.to_csv('prediction.csv', index=False)
