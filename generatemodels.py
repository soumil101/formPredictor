# All the imports
import pandas as pd
import statistics
from sklearn.metrics import pairwise_distances
from joblib import dump, load
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

dist_matrix = pairwise_distances(X, metric='euclidean')
triu_indices = np.triu_indices_from(dist_matrix, k=1)
pairwise_dists = dist_matrix[triu_indices]
mean_distance = np.mean(pairwise_dists)
min_distance = np.min(pairwise_dists)
max_distance = np.max(pairwise_dists)
std_distance = np.std(pairwise_dists)

print("Mean Pairwise Distance:", mean_distance)
print("Minimum Pairwise Distance:", min_distance)
print("Maximum Pairwise Distance:", max_distance)
print("Standard Deviation of Distances:", std_distance)
knn_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn_euclidean.fit(X, y)
knn_manhattan = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn_manhattan.fit(X, y)
knn_chebyshev = KNeighborsClassifier(n_neighbors=1, metric='chebyshev')
knn_chebyshev.fit(X, y)

def give_prediction(test_array):
  options = []
  chosenList = []
  prediction1 = knn_chebyshev.predict(test_array)
  prediction2 = knn_euclidean.predict(test_array)
  prediction3 = knn_manhattan.predict(test_array)
  if prediction1[0] == prediction2[0] and prediction1[0] != prediction3[0]:
    options.append(prediction1[0])
    chosenList.append("chebyshev")
    chosenList.append("euclidean")
  elif prediction1[0] == prediction3[0] and prediction1[0] != prediction2[0]:
    options.append(prediction1[0])
    chosenList.append("chebyshev")
    chosenList.append("manhattan")
  elif prediction2[0] == prediction3[0] and prediction1[0] != prediction2[0]:
    options.append(prediction2[0])
    chosenList.append("euclidean")
    chosenList.append("manhattan")
  else:
    options.append(prediction2[0])
    chosenList.append("euclidean")
  return options[0], chosenList

def calculate_distance_metric(test_array, X, chosenList):
  distances_euclidean = pairwise_distances(test_array,  X, metric='euclidean')
  distances_manhattan = pairwise_distances(test_array, X, metric='manhattan')
  distances_chebyshev = pairwise_distances(test_array, X, metric='chebyshev')

  min_euc = np.min(distances_euclidean)
  min_man = np.min(distances_manhattan)
  min_che = np.min(distances_chebyshev)

  mean_euc = np.mean(distances_euclidean)
  mean_man = np.mean(distances_manhattan)
  mean_che = np.mean(distances_chebyshev)

  meanList = []
  for i in chosenList:
    if i == "euclidean":
      meanList.append(min_euc/mean_euc)
    elif i == "manhattan":
      meanList.append(min_man/mean_man)
    elif i == "chebyshev":
      meanList.append(min_che/mean_che) 
  
  meanSum = 0
  for i in meanList:
    meanSum += i
  meanSum /= len(meanList)
  return meanSum
chris = [[1.0, 0.0], [0.694989106753813, 0.08669454008853898], [0.44589687726942623, 0.17338908017707821], [0.2628903413217139, 0.249754058042302], [0.140885984023239, 0.3146827348745695], [0.05374001452432821, 0.38108706345302507], [0.007262164124909234, 0.44749139203148053], [0.0, 0.5126660108214461], [0.01960784313725487, 0.5742744712247909], [0.0653594771241829, 0.6309640924741761], [0.1960784313725489, 0.6844564682734874], [0.39288307915758897, 0.7283571077225773], [0.5933188090050835, 0.7741023118544024], [0.7785039941902687, 0.8202164289227742], [0.9317356572258533, 0.8622725036891293], [0.9426289034132173, 0.8991637973438269], [0.9448075526506899, 0.9331037875061485], [0.9389978213507626, 0.9613871126414166], [0.9135802469135803, 0.9833989178553861], [0.8482207697893972, 1.0]]
clo = np.array(chris).flatten()
test = np.hstack((clo, calculate_metrics(chris)[0], calculate_metrics(chris)[1], calculate_metrics(chris)[2], calculate_metrics(chris)[3], calculate_metrics(chris)[4], calculate_metrics(chris)[5], calculate_metrics(chris)[6]))
print(calculate_distance_metric([test], X, give_prediction([test])[1]))
dump(knn_manhattan, 'models/knn_manhattan.pkl')
dump(knn_euclidean, 'models/knn_euclidean.pkl')
dump(knn_chebyshev, 'models/knn_chebyshev.pkl')




