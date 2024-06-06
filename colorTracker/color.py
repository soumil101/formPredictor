# Importing necessary folders
import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Video
#cap = cv2.VideoCapture('/Users/vikramkarmarkar/Desktop/School Work/ECS 170 - Spring 2024/Project/formPredicter/colorTracker/kar.png')
hsvVals = 'red'

myColorFinder = ColorFinder(True)
hsvVals = 'red'


while True:
    img = cv2.imread('/Users/vikramkarmarkar/Desktop/School Work/ECS 170 - Spring 2024/Project/formPredicter/colorTracker/kar.png')
    
    imgColor, mask = myColorFinder.update(img, hsvVals)
    
    img = cv2.resize(img, (0,0), None, 0.7, 0.7)
    cv2.imshow("image Color", imgColor)
    
    cv2.waitKey(50)
    

# Reads a single image, processes it to find specific color, displays result
# - continuously reads kar.png from specified path