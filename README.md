# formPredicter
Project that finds the relationship between a real-time shot form with NBA players!

# Data sources:
- https://www.kaggle.com/datasets/paultimothymooney/nba-player-shooting-motions?select=player_metrics.csv
- https://www.kaggle.com/datasets/paultimothymooney/nba-player-shooting-motions?select=path_detail.csv




# distance_to_camera.py 

gets the displacement in 3 dimensions, uses width of object to build a pixel correlation.

Currently set for phone tracking, change `cell_phone` to `basketball` for basketball (I didn't have a basketball)

The file uses `KNOWN_DISTANCE` and `KNOWN_WIDTH` to estimate displacement and displacement from origin. Translates the `KNOWN_DISTANCE` to a length of pixels, then uses that proportion to estimate distance.

`KNOWN_DISTANCE` is 11 inches from camera, `KNOWN_WIDTH` is 2.8 (width of phone).

To get the most accurate distance and displacement predictions, the object has to start at a distance of `KNOWN_DISTANCE` away from the camera. Otherwise, the values will be off

Since `KNOWN_DISTANCE` is 11, you need to hold up the phone 11 inches from the camera and then run `python3 distance_to_camera.py` or `distance_to_camera.py`. After it recognizes the object, and you confirm that the distance from the camera is roughly 0.92 feet, then you can start moving it around.

If we want to modify this to track a ball, we have to **(1)** change `cell_phone` to `basketball`, and **(2)** adjust the `KNOWN_DISTANCE` (maybe 1 foot away initially) and `KNOWN_WIDTH` (width of basketball). You need to hold the ball at `KNOWN_DISTANCE` inches away and then run the command to execute file.

