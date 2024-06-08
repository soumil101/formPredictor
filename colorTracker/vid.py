import os
import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler


def min_max_scaling(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(x - min_val) / (max_val - min_val) for x in lst]

def calculate_points_and_plot(video_path):
    # Initialize the Video
    cap = cv2.VideoCapture(video_path)
    myColorFinder = ColorFinder(False)
    #library HSV: hsvVals = {'hmin': 0, 'smin': 137, 'vmin': 87, 'hmax': 89, 'smax': 255, 'vmax': 255}
    #aadhi arc shot: hsvVals = {'hmin': 0, 'smin': 133, 'vmin': 96, 'hmax': 111, 'smax': 255, 'vmax': 255}
    #(arc3) hsvVals = {'hmin': 0, 'smin': 60, 'vmin': 179, 'hmax': 179, 'smax': 255, 'vmax': 255}
    #(arc4) hsvVals = {'hmin': 0, 'smin': 94, 'vmin': 95, 'hmax': 31, 'smax': 255, 'vmax': 255}
    #lib- vik hsvVals = {'hmin': 0, 'smin': 75, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}
    #chris-lib
    hsvVals = {'hmin': 0, 'smin': 130, 'vmin': 0, 'hmax': 179, 'smax': 255, 'vmax': 255}

    # Variables
    originalPosX = []
    originalPosY = []
    adjustedPosX = []
    adjustedPosY = []
    frame_counter = 0

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
            
            # Polynomial Regression    
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
        
        # Display
        imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
        cv2.imshow("Image", imgContours)
        cv2.waitKey(100)
        
        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

    # Interpolate to get 20 points
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

    # Normalize and stringified X and Y arrays
    normalizedAdjustedPosX = min_max_scaling([1 * x for x in standardized_adjustedPosX])
    normalizedAdjustedPosY = min_max_scaling([-1 * y for y in standardized_adjustedPosY])

    normalizedAdjustedPosX.reverse()
    normalizedAdjustedPosY.reverse()

    normalizedAndStringifiedAdjustedPosX = ' '.join(map(str, normalizedAdjustedPosX))
    normalizedAndStringifiedAdjustedPosY = ' '.join(map(str, normalizedAdjustedPosY))

    averageNormalizedAdjustedPosX = sum(normalizedAdjustedPosX) / len(normalizedAdjustedPosX) if normalizedAdjustedPosX else 0
    averageNormalizedAdjustedPosY = sum(normalizedAdjustedPosY) / len(normalizedAdjustedPosY) if normalizedAdjustedPosY else 0

    # print("\nOriginal Adjusted X Positions:", adjustedPosX, "length:", len(adjustedPosX), "\n")
    # print("Original Adjusted Y Positions:", adjustedPosY, "length:", len(adjustedPosY), "\n")
    print("Normalized Adjusted X Positions (stringified):", normalizedAndStringifiedAdjustedPosX, "\n")
    print("Normalized Adjusted Y Positions (stringified):", normalizedAndStringifiedAdjustedPosY, "\n")
    print("Normalized Adjusted X Positions (UNSTRINGIFIED):", normalizedAdjustedPosX, "\n")
    print("Normalized Adjusted Y Positions (UNSTRINGIFIED):", normalizedAdjustedPosY, "\n")
    print("Average Normalized Adjusted X Position:", averageNormalizedAdjustedPosX, "\n")
    print("Average Normalized Adjusted Y Position:", averageNormalizedAdjustedPosY, "\n")

    print("\n\n\nNormalized, Adjusted, PosX:", normalizedAdjustedPosX, "length:", len(normalizedAdjustedPosX), "\n")
    print("\n\n\nNormalized, Adjusted, PosY:", normalizedAdjustedPosY, "length:", len(normalizedAdjustedPosY), "\n")
    formatted_array = [[normalizedAdjustedPosX[i], normalizedAdjustedPosY[i]] for i in range(len(normalizedAdjustedPosY))]
    print("FORMATTED ARRAY", formatted_array)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot([-1 * x for x in normalizedAdjustedPosX], normalizedAdjustedPosY, marker='o', linestyle='-', color='b')    
    plt.xlabel('Normalized Adjusted X Position')
    plt.ylabel('Normalized Adjusted Y Position')
    plt.title('Normalized Adjusted X vs. Y Position of the Ball')
    plt.grid(True)
    plt.show()

# Run the script
video_path = '/Users/vikramkarmarkar/Desktop/School Work/ECS 170 - Spring 2024/Project/formPredicter/colorTracker/shot-videos/aaron.MOV'
calculate_points_and_plot(video_path)

