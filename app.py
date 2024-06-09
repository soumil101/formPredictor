import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import cv2
import math
import cvzone
from cvzone.ColorModule import ColorFinder
from scipy.interpolate import interp1d
from joblib import load
import tempfile
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from nltk.corpus import wordnet
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import statistics

kmeans_chebyshev = load('models/knn_chebyshev.pkl')
kmeans_euclidean = load('models/knn_euclidean.pkl')
kmeans_manhattan = load('models/knn_manhattan.pkl')

player_pic_dict = {
    'Kobe Bryant': 'kobe.jpeg',
    'Dirk Nowitzki': 'dirk.jpeg',
    'LeBron James': 'james.jpeg',
    'Stephen Curry': 'curry.jpeg',
    'Kevin Durant': 'durant.jpeg',
    'Kawhi Leonard': 'kawhi.jpeg',
    'James Harden': 'harden.jpeg',
    'Damian Lillard': 'dame.jpeg',
    'Jimmy Butler': 'butler.jpeg',
    'Klay Thompson': 'klay.jpeg',
    'J.R Smith': 'smith.jpeg',
    'Carmelo Anthony': 'carmelo.jpeg',
    'Russell Westbrook': 'westbrook.jpeg',
    'Kyrie Irving': 'kyrie.jpeg',
    'Paul George': 'george.jpeg',
    'Chris Paul': 'paul.jpeg',
    'Jeremy Lin': 'lin.jpeg',
    'Draymond Green': 'draymond.jpeg',
    'Danny Green': 'green.jpeg',
    'Derrick Rose': 'rose.jpeg',
    'JJ Redick': 'redick.jpeg',
    'Kyle Korver': 'korver.jpeg',
    'Andre Igoudala': 'iggy.jpeg',
    'Marcus Smart': 'smart.jpeg',
    'Manu Ginobili': 'manu.jpeg',
}

player_names = [
    'Kobe Bryant', 'Dirk Nowitzki', 'Manu Ginobili', 'LeBron James', 'Carmelo Anthony',
    'Kyle Korver', 'Andre Iguodala', 'JR Smith', 'Chris Paul', 'JJ Redick',
    'Kevin Durant', 'Derrick Rose', 'Russell Westbrook', 'James Harden', 'Stephen Curry',
    'Danny Green', 'Paul George', 'Jeremy Lin', 'Kyrie Irving', 'Klay Thompson',
    'Kawhi Leonard', 'Jimmy Butler', 'Damian Lillard', 'Draymond Green', 'Marcus Smart'
]


transformer_manhattan = FunctionTransformer(lambda x: np.sum(np.abs(x), axis=1).reshape(-1, 1), validate=True)
transformer_euclidean = FunctionTransformer(lambda x: x, validate=True)
transformer_chebyshev = FunctionTransformer(lambda x: np.max(np.abs(x), axis=1).reshape(-1, 1), validate=True)

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
  return player_names[options[0]]

def min_max_scaling(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(x - min_val) / (max_val - min_val) for x in lst]

def calculate_points_and_plot(mov):
    cap = cv2.VideoCapture(mov)
    myColorFinder = ColorFinder(False)
    hsvVals = {'hmin': 0, 'smin': 130, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}

    originalPosX = []
    originalPosY = []
    adjustedPosX = []
    adjustedPosY = []

    while True:
        success, img = cap.read()
        if not success:
            break
        
        frameHeight, frameWidth, _ = img.shape
        centerX = frameWidth // 2
        centerY = frameHeight // 2
        
        imgColor, mask = myColorFinder.update(img, hsvVals)
        imgContours, contours = cvzone.findContours(img, mask, minArea=200)
        
        if contours:
            originalX = contours[0]['center'][0]
            originalY = contours[0]['center'][1]
            originalPosX.append(originalX)
            originalPosY.append(originalY)
            
            adjX = originalX - centerX
            adjY = centerY - originalY
            adjustedPosX.append(adjX)
            adjustedPosY.append(adjY)
            
            if adjustedPosX:
                A, B, C = np.polyfit(adjustedPosX, adjustedPosY, 2)
                for i, (posX, posY) in enumerate(zip(adjustedPosX, adjustedPosY)):
                    cv2.circle(imgContours, (posX + centerX, centerY - posY), 10, (0, 255, 0), cv2.FILLED)
                    if i != 0:
                        cv2.line(imgContours, (posX + centerX, centerY - posY), 
                                 (adjustedPosX[i-1] + centerX, centerY - adjustedPosY[i-1]), (0, 255, 0), 5)
                
            for posX, posY in zip(adjustedPosX, adjustedPosY):
                text = f"({posX}, {posY})"
                cv2.putText(imgContours, text, (posX + centerX, centerY - posY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    num_points = 20
    if len(adjustedPosX) > 1:
        interp_func_x = interp1d(range(len(adjustedPosX)), adjustedPosX, kind='linear', fill_value="extrapolate")
        interp_func_y = interp1d(range(len(adjustedPosY)), adjustedPosY, kind='linear', fill_value="extrapolate")
        standardized_indices = np.linspace(0, len(adjustedPosX) - 1, num=num_points)
        standardized_adjustedPosX = interp_func_x(standardized_indices)
        standardized_adjustedPosY = interp_func_y(standardized_indices)
    else:
        standardized_adjustedPosX = adjustedPosX
        standardized_adjustedPosY = adjustedPosY

    normalizedAdjustedPosX = min_max_scaling([1 * x for x in standardized_adjustedPosX])
    normalizedAdjustedPosY = min_max_scaling([-1 * y for y in standardized_adjustedPosY])

    normalizedAdjustedPosX.reverse()
    normalizedAdjustedPosY.reverse()

    normalizedAndStringifiedAdjustedPosX = ' '.join(map(str, normalizedAdjustedPosX))
    normalizedAndStringifiedAdjustedPosY = ' '.join(map(str, normalizedAdjustedPosY))

    averageNormalizedAdjustedPosX = sum(normalizedAdjustedPosX) / len(normalizedAdjustedPosX) if normalizedAdjustedPosX else 0
    averageNormalizedAdjustedPosY = sum(normalizedAdjustedPosY) / len(normalizedAdjustedPosY) if normalizedAdjustedPosY else 0

    formatted_array = [[normalizedAdjustedPosX[i], normalizedAdjustedPosY[i]] for i in range(len(normalizedAdjustedPosY))]

    plt.figure(figsize=(10, 6))
    plt.plot([-1 * x for x in normalizedAdjustedPosX], normalizedAdjustedPosY, marker='o', linestyle='-', color='b')    
    plt.xlabel('Normalized Adjusted X Position')
    plt.ylabel('Normalized Adjusted Y Position')
    plt.title('Normalized Adjusted X vs. Y Position of the Ball')
    plt.grid(True)
    plt.show()

    return formatted_array

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

def process_data(path_detail_df, player_metrics_df):
    def take_every_third_and_limit(x):
        return x[::15][:20] 

    agg_path_detail_df = path_detail_df.groupby('pid').agg({
        'cy': lambda x: take_every_third_and_limit(list(x)),
        'cz': lambda x: take_every_third_and_limit(list(x)),
        'fnm': 'first',
        'lnm': 'first'
    }).reset_index()

    condition = agg_path_detail_df['pid'].isin([201935, 203081, 201142, 201939, 202691, 202681])
    agg_path_detail_df = agg_path_detail_df.loc[condition]

    scaler = MinMaxScaler()
    scaled_rows = []

    for i in range(len(agg_path_detail_df)):
        row = agg_path_detail_df.iloc[i]
        scaled_row = row.copy()

        for col in ['cy', 'cz']:
            if isinstance(row[col], list):
                data_array = np.array(row[col]).reshape(-1, 1)
                scaled_data = scaler.fit_transform(data_array)
                scaled_row[col] = scaled_data.flatten().tolist()

        scaled_rows.append(scaled_row)

    scaled_path_detail_df = pd.DataFrame(scaled_rows, columns=agg_path_detail_df.columns)
    scaled_path_detail_df['full_name'] = scaled_path_detail_df['fnm'] + ' ' + scaled_path_detail_df['lnm']
    scaled_path_detail_df = scaled_path_detail_df.drop(columns=['fnm', 'lnm'])
    scaled_path_detail_df['cy_str'] = scaled_path_detail_df['cy'].apply(lambda x: ' '.join(map(str, x)))
    scaled_path_detail_df['cz_str'] = scaled_path_detail_df['cz'].apply(lambda x: ' '.join(map(str, x)))
    scaled_path_detail_df['cy_mean'] = scaled_path_detail_df['cy'].apply(lambda x: np.mean(x))
    scaled_path_detail_df['cz_mean'] = scaled_path_detail_df['cz'].apply(lambda x: np.mean(x))
    scaled_path_detail_df = scaled_path_detail_df.drop(columns=['cy', 'cz'])

    repetitions = 50
    repeated_df = scaled_path_detail_df.loc[scaled_path_detail_df.index.repeat(repetitions)].reset_index(drop=True)
    return repeated_df, scaled_path_detail_df

def train_model(repeated_df):
    X = repeated_df[['cy_str', 'cz_str', 'cy_mean', 'cz_mean']]
    y = repeated_df['full_name']

    x_dev, x_val, y_dev, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    ros = RandomOverSampler(random_state=42)
    x_os, y_os = ros.fit_resample(x_dev, y_dev)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.2, random_state=42)

    features = FeatureUnion([
        ('cy_str', make_pipeline(TextSelector(key='cy_str'), CountVectorizer())),
        ('cz_str', make_pipeline(TextSelector(key='cz_str'), CountVectorizer())),
        ('cy_mean', NumberSelector(key='cy_mean')),
        ('cz_mean', NumberSelector(key='cz_mean'))
    ])

    pipeline = make_pipeline(
        features,
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    )

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return pipeline, accuracy, classification_report(y_test, y_pred)

def plot_shots(player_df, player_name1, player_name2=None):
    cy1 = player_df[player_df['full_name'] == player_name1]['cy_str'].values[0]
    cz1 = player_df[player_df['full_name'] == player_name1]['cz_str'].values[0]

    cy1 = list(map(float, cy1.split()))
    cz1 = list(map(float, cz1.split()))

    plt.figure(figsize=(5, 3))
    plt.plot(cz1, cy1, marker='o', linestyle='-', color='blue', label=player_name1)

    if player_name2:
        cy2 = player_df[player_df['full_name'] == player_name2]['cy_str'].values[0]
        cz2 = player_df[player_df['full_name'] == player_name2]['cz_str'].values[0]

        cy2 = list(map(float, cy2.split()))
        cz2 = list(map(float, cz2.split()))

        plt.plot(cz2, cy2, marker='s', linestyle='--', color='green', label=player_name2)

    plt.xlabel('Vertical Shot Movement (ft)')
    plt.ylabel('Horizontal Shot Movement (ft)')
    plt.title('Shot Movement Comparison')
    plt.legend()
    st.pyplot(plt)

def process_video(uploaded_video):
    video_bytes = uploaded_video.read()
    video = cv2.VideoCapture(video_bytes)
    _, frame = video.read()
    video.release()
    return frame

def main():
    st.title("Form Finder")

    st.write("Welcome to Form Finder! This app helps you analyze basketball shooting form. We have two core features: shot movement comparison and who do you shoot like? Try both below!")

    st.write("Group 7: Aadhi Aravind, Chris Lo, Kalyan Suvarna, Rahul Prabhu, Soumil Gad, Vikram Choudhry, Vikram Karmarkar")

    path_detail_df = pd.read_csv('Shooting Form Data/path_detail.csv')
    player_metrics_df = pd.read_csv('Shooting Form Data/player_metrics.csv')

    repeated_df, scaled_path_detail_df = process_data(path_detail_df, player_metrics_df)

    # st.write("Our model...")
    # model, accuracy, report = train_model(repeated_df)

    # st.write(f"Model Accuracy: {accuracy:.2f}")
    # st.write("Classification Report:")
    # st.text(report)

    st.title("Player Shot Movement Comparison")
    st.write("Select a player to view shot graphs:")
    players = scaled_path_detail_df['full_name'].unique()
    selected_player1 = st.selectbox("Select First Player", players)
    
    add_second_player = st.checkbox("Select another player for comparison")
    
    if add_second_player:
        selected_player2 = st.selectbox("Select Second Player", players, index=1)
    else:
        selected_player2 = None

    if selected_player1:
        st.write(f"Displaying shot movement for {selected_player1}")
        if selected_player2:
            st.write(f"Comparing with {selected_player2}")
        plot_shots(scaled_path_detail_df, selected_player1, selected_player2)

    st.title("Who do you shoot like?")
    st.write("Upload a video of your shot and we'll tell you who you shoot like! Please shoot a video with an orange basketball and a white background. Your non-dominant hand should be away from the ball. ")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mov") as tmp_file:
            tmp_file.write(uploaded_video.read())
            temp_file_path = tmp_file.name
        
        arr = calculate_points_and_plot(temp_file_path)
        l = np.array(arr).flatten()
        test = np.hstack((l, calculate_metrics(arr)[0], calculate_metrics(arr)[1], calculate_metrics(arr)[2], calculate_metrics(arr)[3], calculate_metrics(arr)[4], calculate_metrics(arr)[5], calculate_metrics(arr)[6]))
        res = give_prediction([test])
        st.image(f"pics/{player_pic_dict[res]}", use_column_width='always')
        st.title(f"You shoot like {res}")

if __name__ == "__main__":
    main()
