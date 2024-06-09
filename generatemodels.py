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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from nltk.corpus import wordnet
import random
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')
path_detail_df = pd.read_csv("Shooting Form Data/path_detail.csv")
players_ids = path_detail_df['pid'].unique()
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
pids_to_use = [201935, 203081, 2594, 2747, 201980, 200755, 201142, 201939, 202331, 2544, 1717, 101108, 201566, 202681, 202691, 202710, 2546, 202695, 202391, 201565, 977, 203110, 203935, 2738, 1938]
def take_every_third_and_limit(x):
    return x[::15][:20]  # Takes every third element and limits to the first 100 points

agg_path_detail_df = path_detail_df.groupby('pid').agg({
    'cy': lambda x: take_every_third_and_limit(list(x)),
    'cz': lambda x: take_every_third_and_limit(list(x)),
    'fnm': 'first',  # or 'last' or another appropriate aggregation function
    'lnm': 'first'   # or 'last' or another appropriate aggregation function
}).reset_index()

# Filter for specific 'pid' values
condition = agg_path_detail_df['pid'].isin(pids_to_use)
agg_path_detail_df = agg_path_detail_df.loc[condition]
print(agg_path_detail_df)

scaled_rows = []

# Initialize MinMaxScaler
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
    print(li)

def calculate_metrics(data):
  metrics = []
  xmean = 0
  ymean = 0
  x = []
  y = []
  for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1])
    xmean += data[i][0]
    ymean += data[i][1]
  xmean /= len(data)
  ymean /= len(data)
  xstd = statistics.stdev(x)
  ystd = statistics.stdev(y)
  xvar = statistics.variance(x)
  yvar = statistics.variance(y)
  metrics.append(xmean)
  metrics.append(ymean)
  metrics.append(xstd)
  metrics.append(ystd)
  metrics.append(xvar)
  metrics.append(yvar)
  slopes = []
  for i in range(len(data) - 1):
      x1, y1 = data[i]
      x2, y2 = data[i + 1]
      slope = (y2 - y1) / (x2 - x1)
      slopes.append(slope)
  slope_changes = []
  for i in range(len(slopes) - 1):
      slope_change = abs(slopes[i + 1] - slopes[i])
      slope_changes.append(slope_change)
  average_slope_change = np.mean(slope_changes)
  metrics.append(average_slope_change)
  return metrics

columns = ['xy', 'xmean', 'ymean', 'xstd', 'ystd', 'xvar', 'yvar','avgdydx', 'Label']
df = pd.DataFrame(columns=columns)
for i in range(len(final)):
  df.loc[i] = [np.array(final[i]), calculate_metrics(final[i])[0], calculate_metrics(final[i])[1], calculate_metrics(final[i])[2], calculate_metrics(final[i])[3], calculate_metrics(final[i])[4], calculate_metrics(final[i])[5], calculate_metrics(final[i])[6], map_pids_to_player[scaled_path_detail_df.iloc[i]['pid']]]
flattened_xy_features = np.array([arr.flatten() for arr in df['xy'].values])
xmean_features = df['xmean'].values.reshape(-1, 1)
ymean_features = df['ymean'].values.reshape(-1, 1)
xstd_features = df['xstd'].values.reshape(-1, 1)
ystd_features = df['ystd'].values.reshape(-1, 1)
xvar_features = df['xvar'].values.reshape(-1, 1)
yvar_features = df['yvar'].values.reshape(-1, 1)
avgdydx_features = df['avgdydx'].values.reshape(-1, 1)
X = np.hstack((flattened_xy_features, xmean_features, ymean_features, xstd_features, ystd_features, xvar_features, yvar_features, avgdydx_features))
y = df['Label']
transformer_manhattan = FunctionTransformer(lambda x: np.sum(np.abs(x), axis=1).reshape(-1, 1), validate=True)
transformer_euclidean = FunctionTransformer(lambda x: x, validate=True)
transformer_chebyshev = FunctionTransformer(lambda x: np.max(np.abs(x), axis=1).reshape(-1, 1), validate=True)

X_manhattan = transformer_manhattan.fit_transform(X)
X_euclidean = transformer_euclidean.fit_transform(X)
X_chebyshev = transformer_chebyshev.fit_transform(X)

n_clusters = len(np.unique(y)) - 1

kmeans_manhattan = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, algorithm='full', init='k-means++')
kmeans_euclidean = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, algorithm='full', init='k-means++')
kmeans_chebyshev = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, algorithm='full', init='k-means++')

y_pred_manhattan = kmeans_manhattan.fit_predict(X_manhattan)
y_pred_euclidean = kmeans_euclidean.fit_predict(X_euclidean)
y_pred_chebyshev = kmeans_chebyshev.fit_predict(X_chebyshev)

silhouette_avg_manhattan = silhouette_score(X_manhattan, y_pred_manhattan)
silhouette_avg_euclidean = silhouette_score(X_euclidean, y_pred_euclidean)
silhouette_avg_chebyshev = silhouette_score(X_chebyshev, y_pred_chebyshev)

calinski_harabasz_manhattan = calinski_harabasz_score(X_manhattan, y_pred_manhattan)
calinski_harabasz_euclidean = calinski_harabasz_score(X_euclidean, y_pred_euclidean)
calinski_harabasz_chebyshev = calinski_harabasz_score(X_chebyshev, y_pred_chebyshev)

davies_bouldin_manhattan = davies_bouldin_score(X_manhattan, y_pred_manhattan)
davies_bouldin_euclidean = davies_bouldin_score(X_euclidean, y_pred_euclidean)
davies_bouldin_chebyshev = davies_bouldin_score(X_chebyshev, y_pred_chebyshev)


print(f'Silhouette Score for K-Means model with Manhattan distance: {silhouette_avg_manhattan}')
print(f'Silhouette Score for K-Means model with Euclidean distance: {silhouette_avg_euclidean}')
print(f'Silhouette Score for K-Means model with Chebyshev distance: {silhouette_avg_chebyshev}')

print(f'Calinski-Harabasz Score for K-Means model with Manhattan distance: {calinski_harabasz_manhattan}')
print(f'Calinski-Harabasz Score for K-Means model with Euclidean distance: {calinski_harabasz_euclidean}')
print(f'Calinski-Harabasz Score for K-Means model with Chebyshev distance: {calinski_harabasz_chebyshev}')

print(f'Davies-Bouldin Score for K-Means model with Manhattan distance: {davies_bouldin_manhattan}')
print(f'Davies-Bouldin Score for K-Means model with Euclidean distance: {davies_bouldin_euclidean}')
print(f'Davies-Bouldin Score for K-Means model with Chebyshev distance: {davies_bouldin_chebyshev}')

# dump to pickles
dump(kmeans_manhattan, 'models/knn_manhattan.pkl')
dump(kmeans_euclidean, 'models/knn_euclidean.pkl')
dump(kmeans_chebyshev, 'models/knn_chebyshev.pkl')

def give_prediction(test_array):
  options = []
  prediction1 = kmeans_chebyshev.predict(transformer_chebyshev.fit_transform(test_array))
  prediction2 = kmeans_euclidean.predict(test_array)
  prediction3 = kmeans_manhattan.predict(transformer_manhattan.fit_transform(test_array))
  if prediction1[0] == prediction2[0] and prediction1[0] != prediction3[0]:
    options.append(prediction1[0])
  elif prediction1[0] == prediction3[0] and prediction1[0] != prediction2[0]:
    options.append(prediction1[0])
  elif prediction2[0] == prediction3[0] and prediction1[0] != prediction2[0]:
    options.append(prediction2[0])
  else:
    options.append(prediction2[0])
  return y[options[0]]