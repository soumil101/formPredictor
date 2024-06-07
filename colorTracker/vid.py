# import os
# import json
# import math
# import cv2
# import cvzone
# from cvzone.ColorModule import ColorFinder
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# def transform_coordinates(json_path, output_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     # Get image dimensions
#     imageWidth = data['imageWidth']
#     imageHeight = data['imageHeight']
    
#     # Calculate center of the image
#     centerX = imageWidth / 2
#     centerY = imageHeight / 2
    
#     shape = data['shapes'][0]
#     x1, y1 = shape['points'][0]
#     x2, y2 = shape['points'][1]
    
#     # Transform the coordinates
#     transformed_x1 = x1 - centerX
#     transformed_y1 = centerY - y1
#     transformed_x2 = x2 - centerX
#     transformed_y2 = centerY - y2
    
#     transformed_shape = {
#         "label": shape['label'],
#         "points": [
#             [transformed_x1, transformed_y1],
#             [transformed_x2, transformed_y2]
#         ],
#         "shape_type": shape['shape_type']
#     }
    
#     transformed_data = {
#         "version": data['version'],
#         "flags": data['flags'],
#         "shapes": [transformed_shape],
#         "imagePath": data['imagePath'],
#         "imageHeight": data['imageHeight'],
#         "imageWidth": data['imageWidth']
#     }
    
#     with open(output_path, 'w') as f:
#         json.dump(transformed_data, f, indent=4)

# def load_transformed_ground_truth(frame_number):
#     json_path = f"../adjusted-bounding-box-coords/frame_{frame_number}_transformed.json"
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     shape = data['shapes'][0]
#     x1, y1 = shape['points'][0]
#     x2, y2 = shape['points'][1]
    
#     return [x1, y1, x2, y2]

# def calculate_iou(boxA, boxB):
#     # Ensure coordinates are valid
#     boxA = [min(boxA[0], boxA[2]), min(boxA[1], boxA[3]), max(boxA[0], boxA[2]), max(boxA[1], boxA[3])]
#     boxB = [min(boxB[0], boxB[2]), min(boxB[1], boxB[3]), max(boxB[0], boxB[2]), max(boxB[1], boxB[3])]
    
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
    
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
#     # Debug prints
#     print(f"boxA: {boxA}, boxB: {boxB}")
#     print(f"xA: {xA}, yA: {yA}, xB: {xB}, yB: {yB}")
#     print(f"interArea: {interArea}, boxAArea: {boxAArea}, boxBArea: {boxBArea}")

#     if (boxAArea + boxBArea - interArea) == 0:
#         return 0
    
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou



# input_json_dir = '../frames-from-shot-vid'
# output_json_dir = '../adjusted-bounding-box-coords'
# if not os.path.exists(output_json_dir):
#     os.makedirs(output_json_dir)

# for filename in os.listdir(input_json_dir):
#     if filename.endswith('.json'):
#         input_path = os.path.join(input_json_dir, filename)
#         output_path = os.path.join(output_json_dir, filename.replace('.json', '_transformed.json'))
#         transform_coordinates(input_path, output_path)

# # Create a directory to save frames if it does not exist
# frames_dir = 'frames'
# if not os.path.exists(frames_dir):
#     os.makedirs(frames_dir)

# # Initialize the Video
# cap = cv2.VideoCapture('/Users/vikramkarmarkar/Desktop/School Work/ECS 170 - Spring 2024/Project/formPredicter/colorTracker/aadhi.MOV')

# # Create the Color Finder Object (False to not run the slider)
# myColorFinder = ColorFinder(False)
# hsvVals = {'hmin': 0, 'smin': 137, 'vmin': 87, 'hmax': 89, 'smax': 255, 'vmax': 255}

# # Variables
# originalPosX = []
# originalPosY = []
# adjustedPosX = []
# adjustedPosY = []
# xList = [item for item in range(0, 1300)]
# prediction = False

# frame_counter = 0  # Initialize frame counter
# iou_list = []  # List to store IoU values

# while True:
#     success, img = cap.read()
    
#     if not success:
#         break
    
#     # Get the dimensions of the frame
#     frameHeight, frameWidth, _ = img.shape
#     centerX = frameWidth // 2
#     centerY = frameHeight // 2
    
#     # Find the Color of the Ball
#     imgColor, mask = myColorFinder.update(img, hsvVals)
    
#     # Find location of the ball
#     imgContours, contours = cvzone.findContours(img, mask, minArea=200)
    
#     # Checking if item exists, if it does get the center of the item
#     if contours:
#         originalX = contours[0]['center'][0]
#         originalY = contours[0]['center'][1]
#         originalPosX.append(originalX)
#         originalPosY.append(originalY)
        
#         # Adjust coordinates
#         adjX = originalX - centerX
#         adjY = centerY - originalY
#         adjustedPosX.append(adjX)
#         adjustedPosY.append(adjY)
        
#         # Save the current frame to the frames directory
#         frame_filename = os.path.join(frames_dir, f'frame_{frame_counter}.jpg')
#         cv2.imwrite(frame_filename, img)
        
#         # Calculate IoU with ground truth
#         ground_truth_box = load_transformed_ground_truth(frame_counter)
#         # Transform detected box coordinates to have the origin centered at (0,0)
#         detected_box = [contours[0]['bbox'][0] - centerX, 
#                         centerY - contours[0]['bbox'][1], 
#                         contours[0]['bbox'][0] + contours[0]['bbox'][2] - centerX, 
#                         centerY - (contours[0]['bbox'][1] + contours[0]['bbox'][3])]

#         print(f"Frame {frame_counter}:")
#         print(f"Ground Truth Box: {ground_truth_box}")
#         print(f"Detected Box: {detected_box}")

#         iou = calculate_iou(ground_truth_box, detected_box)
#         print(f"IoU: {iou}")
#         iou_list.append(iou)

#         frame_counter += 1

        
#     # Polynomial Regression    
#     if adjustedPosX:
#         # Find the Coefficients
#         A, B, C = np.polyfit(adjustedPosX, adjustedPosY, 2)
        
#         # For loop to make the dots and line trail
#         for i, (posX, posY) in enumerate(zip(adjustedPosX, adjustedPosY)):
#             cv2.circle(imgContours, (posX + centerX, centerY - posY), 10, (0, 255, 0), cv2.FILLED)
#             if i != 0:
#                 cv2.line(imgContours, (posX + centerX, centerY - posY), 
#                          (adjustedPosX[i-1] + centerX, centerY - adjustedPosY[i-1]), (0, 255, 0), 5)
                
#     for posX, posY in zip(adjustedPosX, adjustedPosY):
#         text = f"({posX}, {posY})"
#         cv2.putText(imgContours, text, (posX + centerX, centerY - posY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
#     # Display
#     imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
#     cv2.imshow("Image", imgContours)
#     cv2.waitKey(100)  # Parameter is the amount of time you want the video to play

#     frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps > 0:
#         video_duration = frame_count / fps
#     else:
#         video_duration = 0

# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()

# # ------------------------------
# # Normalized X and Y arrays
# # ------------------------------
# # Min-Max Scaling for adjustedPosX and adjustedPosY
# def min_max_scaling(lst):
#     min_val = min(lst)
#     max_val = max(lst)
#     scaled_lst = [(x - min_val) / (max_val - min_val) for x in lst]
#     return scaled_lst

# # Apply Min-Max Scaling
# normalizedAdjustedPosX = min_max_scaling([1 * x for x in adjustedPosX])
# normalizedAdjustedPosY = min_max_scaling([-1 * y for y in adjustedPosY])

# normalizedAdjustedPosX.reverse()
# normalizedAdjustedPosY.reverse()

# # ------------------------------
# # Normalized AND stringified X and Y arrays
# # ------------------------------
# normalizedAndStringifiedAdjustedPosX = ' '.join(map(str, normalizedAdjustedPosX))
# normalizedAndStringifiedAdjustedPosY = ' '.join(map(str, normalizedAdjustedPosY))

# # Calculate average normalized values
# averageNormalizedAdjustedPosX = sum(normalizedAdjustedPosX) / len(normalizedAdjustedPosX) if normalizedAdjustedPosX else 0
# averageNormalizedAdjustedPosY = sum(normalizedAdjustedPosY) / len(normalizedAdjustedPosY) if normalizedAdjustedPosY else 0

# # Print the arrays and averages
# print("\nOriginal Adjusted X Positions:", adjustedPosX, "length:", len(adjustedPosX), "\n")
# print("Original Adjusted Y Positions:", adjustedPosY, "length:", len(adjustedPosY), "\n")
# print("Normalized Adjusted X Positions (stringified):", normalizedAndStringifiedAdjustedPosX, "\n")
# print("Normalized Adjusted Y Positions (stringified):", normalizedAndStringifiedAdjustedPosY, "\n")
# print("Average Normalized Adjusted X Position:", averageNormalizedAdjustedPosX, "\n")
# print("Average Normalized Adjusted Y Position:", averageNormalizedAdjustedPosY, "\n")
# print("Video duration (seconds):", video_duration, "\n")
# print("Average IoU:", sum(iou_list) / len(iou_list) if iou_list else 0, "\n")

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(normalizedAdjustedPosX, normalizedAdjustedPosY, marker='o', linestyle='-', color='b')

# # Add labels and title
# plt.xlabel('Normalized Adjusted X Position')
# plt.ylabel('Normalized Adjusted Y Position')
# plt.title('Normalized Adjusted X vs. Y Position of the Ball')

# # Display the plot
# plt.grid(True)
# plt.show()




# processes a vid to track the position of a basketball and predicts its trajectory using polynomial reg 
# - initialize vid
# - color finder setup
# - HSV color range
# - Variables for tracking: posListX, posListY, store x and y coordinates of detected basketball over time 
# - process vid, track basketball 
# - visualization of points on the image/vid



import os
import cv2
import json
import cvzone
import numpy as np
import matplotlib.pyplot as plt
from cvzone.ColorModule import ColorFinder
from sklearn.preprocessing import MinMaxScaler

# Part 1: Frame Extraction
def extract_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    cap = cv2.VideoCapture(video_path)
    myColorFinder = ColorFinder(False)
    hsvVals = {'hmin': 0, 'smin': 137, 'vmin': 87, 'hmax': 89, 'smax': 255, 'vmax': 255}
    
    frame_counter = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        
        imgColor, mask = myColorFinder.update(img, hsvVals)
        imgContours, contours = cvzone.findContours(img, mask, minArea=200)
        
        if contours:
            frame_filename = os.path.join(frames_dir, f'frame_{frame_counter}.jpg')
            cv2.imwrite(frame_filename, img)
            frame_counter += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Frames extracted. Please annotate them using your annotation tool.")

# Part 2: Manual Annotation (User manually annotates frames)

# Part 3: IoU Calculation and Post-Processing
def transform_coordinates(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    imageWidth = data['imageWidth']
    imageHeight = data['imageHeight']
    centerX = imageWidth / 2
    centerY = imageHeight / 2
    
    shape = data['shapes'][0]
    x1, y1 = shape['points'][0]
    x2, y2 = shape['points'][1]
    
    transformed_x1 = x1 - centerX
    transformed_y1 = centerY - y1
    transformed_x2 = x2 - centerX
    transformed_y2 = centerY - y2
    
    transformed_shape = {
        "label": shape['label'],
        "points": [
            [transformed_x1, transformed_y1],
            [transformed_x2, transformed_y2]
        ],
        "shape_type": shape['shape_type']
    }
    
    transformed_data = {
        "version": data['version'],
        "flags": data['flags'],
        "shapes": [transformed_shape],
        "imagePath": data['imagePath'],
        "imageHeight": data['imageHeight'],
        "imageWidth": data['imageWidth']
    }
    
    with open(output_path, 'w') as f:
        json.dump(transformed_data, f, indent=4)

def load_transformed_ground_truth(frame_number):
    json_path = f"../adjusted-bounding-box-coords/frame_{frame_number}_transformed.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    shape = data['shapes'][0]
    x1, y1 = shape['points'][0]
    x2, y2 = shape['points'][1]
    
    return [x1, y1, x2, y2]

def calculate_iou(boxA, boxB):
    boxA = [min(boxA[0], boxA[2]), min(boxA[1], boxA[3]), max(boxA[0], boxA[2]), max(boxA[1], boxA[3])]
    boxB = [min(boxB[0], boxB[2]), min(boxB[1], boxB[3]), max(boxB[0], boxB[2]), max(boxB[1], boxB[3])]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if (boxAArea + boxBArea - interArea) == 0:
        return 0
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_iou_and_plot(video_path, frames_dir, input_json_dir, output_json_dir):
    if not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir)
    
    for filename in os.listdir(input_json_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_json_dir, filename)
            output_path = os.path.join(output_json_dir, filename.replace('.json', '_transformed.json'))
            transform_coordinates(input_path, output_path)

    cap = cv2.VideoCapture(video_path)
    myColorFinder = ColorFinder(False)
    hsvVals = {'hmin': 0, 'smin': 137, 'vmin': 87, 'hmax': 89, 'smax': 255, 'vmax': 255}

    originalPosX = []
    originalPosY = []
    adjustedPosX = []
    adjustedPosY = []
    iou_list = []

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
            
            ground_truth_box = load_transformed_ground_truth(frame_counter)
            detected_box = [contours[0]['bbox'][0] - centerX, 
                            centerY - contours[0]['bbox'][1], 
                            contours[0]['bbox'][0] + contours[0]['bbox'][2] - centerX, 
                            centerY - (contours[0]['bbox'][1] + contours[0]['bbox'][3])]

            iou = calculate_iou(ground_truth_box, detected_box)
            iou_list.append(iou)

            frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

    average_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    print("Average IoU:", average_iou)

    normalizedAdjustedPosX = min_max_scaling([1 * x for x in adjustedPosX])
    normalizedAdjustedPosY = min_max_scaling([-1 * y for y in adjustedPosY])

    normalizedAdjustedPosX.reverse()
    normalizedAdjustedPosY.reverse()

    normalizedAndStringifiedAdjustedPosX = ' '.join(map(str, normalizedAdjustedPosX))
    normalizedAndStringifiedAdjustedPosY = ' '.join(map(str, normalizedAdjustedPosY))

    averageNormalizedAdjustedPosX = sum(normalizedAdjustedPosX) / len(normalizedAdjustedPosX) if normalizedAdjustedPosX else 0
    averageNormalizedAdjustedPosY = sum(normalizedAdjustedPosY) / len(normalizedAdjustedPosY) if normalizedAdjustedPosY else 0

    print("\nOriginal Adjusted X Positions:", adjustedPosX, "length:", len(adjustedPosX), "\n")
    print("Original Adjusted Y Positions:", adjustedPosY, "length:", len(adjustedPosY), "\n")
    print("Normalized Adjusted X Positions (stringified):", normalizedAndStringifiedAdjustedPosX, "\n")
    print("Normalized Adjusted Y Positions (stringified):", normalizedAndStringifiedAdjustedPosY, "\n")
    print("Average Normalized Adjusted X Position:", averageNormalizedAdjustedPosX, "\n")
    print("Average Normalized Adjusted Y Position:", averageNormalizedAdjustedPosY, "\n")

    plt.figure(figsize=(10, 6))
    plt.plot(normalizedAdjustedPosX, normalizedAdjustedPosY, marker='o', linestyle='-', color='b')
    plt.xlabel('Normalized Adjusted X Position')
    plt.ylabel('Normalized Adjusted Y Position')
    plt.title('Normalized Adjusted X vs. Y Position of the Ball')
    plt.grid(True)
    plt.show()

def min_max_scaling(lst):
    min_val = min(lst)
    max_val = max(lst)
    return [(x - min_val) / (max_val - min_val) for x in lst]

video_path = '/Users/vikramkarmarkar/Desktop/School Work/ECS 170 - Spring 2024/Project/formPredicter/colorTracker/aadhi.MOV'
frames_dir = 'frames'
input_json_dir = '../frames-from-shot-vid'
output_json_dir = '../adjusted-bounding-box-coords'

# Part 1: Extract frames
extract_frames(video_path, frames_dir)

# Part 2: Manually annotate the frames (this step is manual and done outside the script)

# Part 3: Calculate IoU and perform post-processing
calculate_iou_and_plot(video_path, frames_dir, input_json_dir, output_json_dir)

