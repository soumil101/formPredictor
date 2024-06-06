import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Video
cap = cv2.VideoCapture('/Users/vikramkarmarkar/Desktop/School Work/ECS 170 - Spring 2024/Project/formPredicter/colorTracker/aadhi.MOV')

# Create the Color Finder Object (False to not run the slider)
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 137, 'vmin': 87, 'hmax': 89, 'smax': 255, 'vmax': 255}

# Variables
posListX = []
posListY = []
xList = [item for item in range(0, 1300)]
prediction = False

while True:
    success, img = cap.read()
    
    if not success:
        break
    
    # Find the Color of the Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)
    
    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=200)
    
    # Checking if item exists, if it does get the center of the item
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])
        
    # Polynomial Regression    
    if posListX:
        # Find the Coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)
        
        # For loop to make the dots and line trail
        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            cv2.circle(imgContours, (posX, posY), 10, (0, 255, 0), cv2.FILLED)
            if i != 0:
                cv2.line(imgContours, (posX, posY), (posListX[i-1], posListY[i-1]), (0, 255, 0), 5)
                
    for posX, posY in zip(posListX, posListY):
        text = f"({posX}, {posY})"
        cv2.putText(imgContours, text, (posX, posY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    cv2.imshow("Image", imgContours)
    cv2.waitKey(100)  # Parameter is the amount of time you want the video to play

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        video_duration = frame_count / fps
    else:
        video_duration = 0

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Min-Max Scaling for posListX and posListY
def min_max_scaling(lst):
    min_val = min(lst)
    max_val = max(lst)
    scaled_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return scaled_lst

# Apply Min-Max Scaling
posListX_scaled = min_max_scaling(posListX)
posListY_scaled = min_max_scaling(posListY)

# Print posListX and posListY
# print("posListX:", posListX, "length: ", len(posListX))
# print("posListY:", posListY, "length: ", len(posListY))

# Data for ML Model
posListX_str = ' '.join(map(str, posListX_scaled))
posListY_str = ' '.join(map(str, posListY_scaled))

average_posListX = sum(posListX_scaled) / len(posListX_scaled) if posListX_scaled else 0
average_posListY = sum(posListY_scaled) / len(posListY_scaled) if posListY_scaled else 0

print("\nposListX:", posListX, "length: ", len(posListX), "\n")
print("posListY:", posListY, "length: ", len(posListY), "\n")
print("posListX (scaled and stringified):", posListX_str, "\n")
print("posListY (scaled and stringified):", posListY_str, "\n")
print("Average posX:", average_posListX, "\n")
print("Average posY:", average_posListY, "\n")
print("Video duration (seconds):", video_duration, "\n")




# processes a vid to track the position of a basketball and predicts its trajectory using polynomial reg 
# - initialize vid
# - color finder setup
# - HSV color range
# - Variables for tracking: posListX, posListY, store x and y coordinates of detected basketball over time 
# - process vid, track basketball 
# - visualization of points on the image/vid
