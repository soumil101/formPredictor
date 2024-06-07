import os
import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Create a directory to save frames if it does not exist
frames_dir = 'frames'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Initialize the Video
cap = cv2.VideoCapture('/Users/vikramkarmarkar/Desktop/School Work/ECS 170 - Spring 2024/Project/formPredicter/colorTracker/chris.MOV')

# Create the Color Finder Object (False to not run the slider)
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 137, 'vmin': 87, 'hmax': 89, 'smax': 255, 'vmax': 255}

# Variables
originalPosX = []
originalPosY = []
adjustedPosX = []
adjustedPosY = []
xList = [item for item in range(0, 1300)]
prediction = False

frame_counter = 0  # Initialize frame counter

while True:
    success, img = cap.read()
    
    if not success:
        break
    
    # Get the dimensions of the frame
    frameHeight, frameWidth, _ = img.shape
    centerX = frameWidth // 2
    centerY = frameHeight // 2
    
    # Find the Color of the Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)
    
    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=200)
    
    # Checking if item exists, if it does get the center of the item
    if contours:
        originalX = contours[0]['center'][0]
        originalY = contours[0]['center'][1]
        originalPosX.append(originalX)
        originalPosY.append(originalY)
        
        # Adjust coordinates
        adjX = originalX - centerX
        adjY = centerY - originalY
        adjustedPosX.append(adjX)
        adjustedPosY.append(adjY)
        
        # Save the current frame to the frames directory
        frame_filename = os.path.join(frames_dir, f'frame_{frame_counter}.jpg')
        cv2.imwrite(frame_filename, img)
        frame_counter += 1
        
    # Polynomial Regression    
    if adjustedPosX:
        # Find the Coefficients
        A, B, C = np.polyfit(adjustedPosX, adjustedPosY, 2)
        
        # For loop to make the dots and line trail
        for i, (posX, posY) in enumerate(zip(adjustedPosX, adjustedPosY)):
            cv2.circle(imgContours, (posX + centerX, centerY - posY), 10, (0, 255, 0), cv2.FILLED)
            if i != 0:
                cv2.line(imgContours, (posX + centerX, centerY - posY), 
                         (adjustedPosX[i-1] + centerX, centerY - adjustedPosY[i-1]), (0, 255, 0), 5)
                
    for posX, posY in zip(adjustedPosX, adjustedPosY):
        text = f"({posX}, {posY})"
        cv2.putText(imgContours, text, (posX + centerX, centerY - posY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
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

# ------------------------------
# Normalized X and Y arrays
# ------------------------------
# Min-Max Scaling for adjustedPosX and adjustedPosY
def min_max_scaling(lst):
    min_val = min(lst)
    max_val = max(lst)
    scaled_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return scaled_lst

# Apply Min-Max Scaling


normalizedAdjustedPosX = min_max_scaling([1 * x for x in adjustedPosX])
normalizedAdjustedPosY = min_max_scaling([-1 * y for y in adjustedPosY])

normalizedAdjustedPosX.reverse()
normalizedAdjustedPosY.reverse()

# ------------------------------
# Normalized AND stringified X and Y arrays
# ------------------------------
normalizedAndStringifiedAdjustedPosX = ' '.join(map(str, normalizedAdjustedPosX))
normalizedAndStringifiedAdjustedPosY = ' '.join(map(str, normalizedAdjustedPosY))

# Calculate average normalized values
averageNormalizedAdjustedPosX = sum(normalizedAdjustedPosX) / len(normalizedAdjustedPosX) if normalizedAdjustedPosX else 0
averageNormalizedAdjustedPosY = sum(normalizedAdjustedPosY) / len(normalizedAdjustedPosY) if normalizedAdjustedPosY else 0

# Print the arrays and averages
print("\nOriginal Adjusted X Positions:", adjustedPosX, "length:", len(adjustedPosX), "\n")
print("Original Adjusted Y Positions:", adjustedPosY, "length:", len(adjustedPosY), "\n")
print("Normalized Adjusted X Positions (stringified):", normalizedAndStringifiedAdjustedPosX, "\n")
print("Normalized Adjusted Y Positions (stringified):", normalizedAndStringifiedAdjustedPosY, "\n")
print("Average Normalized Adjusted X Position:", averageNormalizedAdjustedPosX, "\n")
print("Average Normalized Adjusted Y Position:", averageNormalizedAdjustedPosY, "\n")
print("Video duration (seconds):", video_duration, "\n")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(normalizedAdjustedPosX, normalizedAdjustedPosY, marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('Normalized Adjusted X Position')
plt.ylabel('Normalized Adjusted Y Position')
plt.title('Normalized Adjusted X vs. Y Position of the Ball')

# Display the plot
plt.grid(True)
plt.show()




# processes a vid to track the position of a basketball and predicts its trajectory using polynomial reg 
# - initialize vid
# - color finder setup
# - HSV color range
# - Variables for tracking: posListX, posListY, store x and y coordinates of detected basketball over time 
# - process vid, track basketball 
# - visualization of points on the image/vid
