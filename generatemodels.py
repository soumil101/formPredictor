# All the imports
import pandas as pd
import statistics
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import imblearn.over_sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from nltk.corpus import wordnet
import random
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

# Load data
path_detail_df = pd.read_csv("./Shooting Form Data/path_detail.csv")
players_ids = path_detail_df['pid'].unique()

# Map player IDs to names
map_pids_to_player = {}
for i in range(len(players_ids)):
    filtered_df = path_detail_df[path_detail_df['pid'] == players_ids[i]]
    other_column1_values = filtered_df['fnm'].tolist()
    other_column2_values = filtered_df['lnm'].tolist()
    s = ""
    s += other_column1_values[0]
    s += " "
    s += other_column2_values[0]
    map_pids_to_player[players_ids[i]] = s

# Specify player IDs to use
pids_to_use = [201935, 203081, 2594, 2747, 201980, 200755, 201142, 201939, 202331, 2544, 1717, 101108, 201566, 202681, 202691, 202710, 2546, 202695, 202391, 201565, 977, 203110, 203935, 2738, 1938]

# Function to take every third element and limit
def take_every_third_and_limit(x):
    return x[::15][:20]

# Aggregate data
agg_path_detail_df = path_detail_df.groupby('pid').agg({
    'cy': lambda x: take_every_third_and_limit(list(x)),
    'cz': lambda x: take_every_third_and_limit(list(x)),
    'fnm': 'first',
    'lnm': 'first'
}).reset_index()

# Filter for specific 'pid' values
condition = agg_path_detail_df['pid'].isin(pids_to_use)
agg_path_detail_df = agg_path_detail_df.loc[condition]

# Scale data
scaled_rows = []
scaler = MinMaxScaler()
for i in range(len(agg_path_detail_df)):
    row = agg_path_detail_df.iloc[i]
    scaled_row = row.copy()
    for col in ['cy', 'cz']:
        try:
            if isinstance(row[col], list):
                data_array = np.array(row[col]).reshape(-1, 1)
                scaled_data = scaler.fit_transform(data_array)
                scaled_row[col] = scaled_data.flatten().tolist()
            else:
                print(f"Skipping row {i} for column {col} as it is not a list.")
        except Exception as e:
            raise
    scaled_rows.append(scaled_row)
scaled_path_detail_df = pd.DataFrame(scaled_rows, columns=agg_path_detail_df.columns)

# Prepare data for KNN
final = []
for i in range(len(scaled_path_detail_df)):
    li = []
    print("Player Name:", map_pids_to_player[scaled_path_detail_df.iloc[i]['pid']])
    for x in range(len(scaled_path_detail_df.iloc[i]['cy'])):
        temp = []
        temp.append(scaled_path_detail_df.iloc[i]['cy'][x])
        temp.append(scaled_path_detail_df.iloc[i]['cz'][x])
        li.append(temp)
    final.append(li)

players_database = []
for x in range(len(scaled_path_detail_df)):
    cy = scaled_path_detail_df.iloc[x]['cy']
    cz = scaled_path_detail_df.iloc[x]['cz']
    formatted_array = [[cy[i], cz[i]] for i in range(len(cy))]
    players_database.append(formatted_array)
players_database = np.array(players_database)

labels = []
for i in range(len(scaled_path_detail_df)):
    labels.append(map_pids_to_player[scaled_path_detail_df.iloc[i]['pid']])
labeled_arrays = [(labels[i], pd.DataFrame(players_database[i])) for i in range(len(players_database))]

final_df = pd.concat([df.assign(Label=label) for label, df in labeled_arrays])
X = final_df.drop('Label', axis=1)
y = final_df['Label']

# Define KNN models
knn_chebyshev = KNeighborsClassifier(n_neighbors=1, metric='chebyshev')
knn_chebyshev.fit(X, y)

knn_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn_euclidean.fit(X, y)

knn_manhattan = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn_manhattan.fit(X, y)

# Save the models
dump(knn_chebyshev, 'models/knn_chebyshev.pkl')
dump(knn_euclidean, 'models/knn_euclidean.pkl')
dump(knn_manhattan, 'models/knn_manhattan.pkl')

# Prediction function
def give_prediction(test_array):
    options = []
    prediction1 = knn_chebyshev.predict(test_array)
    prediction2 = knn_euclidean.predict(test_array)
    prediction3 = knn_manhattan.predict(test_array)
    
    if prediction1[0] == prediction2[0] == prediction3[0]:
        options.append(prediction1[0])
    else:
        options.append(prediction1[0])
        options.append(prediction2[0])
        options.append(prediction3[0])
    return options
