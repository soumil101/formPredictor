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

#TODO
def process_video(uploaded_video):
    video_bytes = uploaded_video.read()
    video = cv2.VideoCapture(video_bytes)
    _, frame = video.read()
    video.release()
    return frame


def main():
    st.title("Shot Movement Analysis")

    path_detail_df = pd.read_csv('path_detail.csv')
    player_metrics_df = pd.read_csv('player_metrics.csv')

    repeated_df, scaled_path_detail_df = process_data(path_detail_df, player_metrics_df)

    st.write("Our model...")
    model, accuracy, report = train_model(repeated_df)

    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(report)

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

    st.write("Upload a video for analysis:")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        st.write("Processing video...")
        frame = process_video(uploaded_video)
        
        st.write("Video uploaded successfully!")
        st.image(frame, caption="First frame of the uploaded video")

if __name__ == "__main__":
    main()
