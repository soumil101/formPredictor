# Importing necessary folders
import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Video
cap = cv2.VideoCapture('/Users/chrislo/Desktop/Personal Github/BasketballAI---March-2023/Files/Videos/vid (2).mp4')
hsvVals = 'red'

myColorFinder = ColorFinder(True)
hsvVals = 'red'


while True:
    img = cv2.imread('/Users/chrislo/Desktop/Personal Github/BasketballAI---March-2023/Files/vik.png')
    
    imgColor, mask = myColorFinder.update(img, hsvVals)
    
    img = cv2.resize(img, (0,0), None, 0.7, 0.7)
    cv2.imshow("image Color", imgColor)
    
    cv2.waitKey(50)
    