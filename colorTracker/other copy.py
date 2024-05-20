import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default ID for the built-in webcam. Use 1 or another ID for external webcams.

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the ColorFinder
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 130, 'vmin': 132, 'hmax': 179, 'smax': 255, 'vmax': 255}

# Variables
posListX = []
posListY = []
xList = [item for item in range(0, 1300)]
prediction = False

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Crop the image if necessary (adjust according to your requirements)
    if img is None:
        print("Error: Frame is None.")
        continue
    img = img[0:900, :]  # cropping image (y-value, x-value)

    # Find the Color of the Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=200)

    # Checking if item exists, if it does gets the center of the item
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

        # Limit the lists to the last 10 points
        posListX = posListX[-10:]
        posListY = posListY[-10:]

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
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
