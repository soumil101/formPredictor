# formPredicter
Project that finds the relationship between a real-time shot form with NBA players!

# Data sources:
- https://www.kaggle.com/datasets/paultimothymooney/nba-player-shooting-motions?select=player_metrics.csv
- https://www.kaggle.com/datasets/paultimothymooney/nba-player-shooting-motions?select=path_detail.csv

# Streamlit Front end link:
- https://formpredictor.streamlit.app/

# File Organization Walkthrough

- Shooting Form Data
  - Stores the dataset
- colorTracker
  - Files with attempts of colortracking and "vid.py" is the script with the working code
- models
  - pickle files for our models
- pics
  - pictures of the players in our dataset to be outputted
- videos
  - Videos of group members and other friends that we used to test
- KNNMODEL.ipynb
  - notebook for the creation of the models
- Player-Data-Plotting.ipynb
  - notebook for plotting the graphs
- app.py
  - frontend code
- generatemodels.py
  - generate the pickle models
- requirements.txt
  - requirements for streamlit
