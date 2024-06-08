import os
import cv2
import json
import cvzone
from cvzone.ColorModule import ColorFinder

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

# Part 3: IoU Calculation
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

def calculate_iou_for_video(video_path, frames_dir, input_json_dir, output_json_dir):
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
            try:
                ground_truth_box = load_transformed_ground_truth(frame_counter)
                detected_box = [contours[0]['bbox'][0] - centerX, 
                                centerY - contours[0]['bbox'][1], 
                                contours[0]['bbox'][0] + contours[0]['bbox'][2] - centerX, 
                                centerY - (contours[0]['bbox'][1] + contours[0]['bbox'][3])]

                iou = calculate_iou(ground_truth_box, detected_box)
                iou_list.append(iou)

            except FileNotFoundError:
                print(f"Annotation file for frame {frame_counter} not found. Skipping IoU calculation for this frame.")

            frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

    average_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    print("Average IoU:", average_iou)

# Run the script
video_path = '/Users/vikramkarmarkar/Desktop/School Work/ECS 170 - Spring 2024/Project/formPredicter/colorTracker/shot-videos/kalyan_arc.MOV'
frames_dir = 'frames'
input_json_dir = './frames'
output_json_dir = '../adjusted-bounding-box-coords'

# Part 1: Extract frames
extract_frames(video_path, frames_dir)

# Part 2: Manually annotate the frames (this step is manual and done outside the script)

# Part 3: Calculate IoU
calculate_iou_for_video(video_path, frames_dir, input_json_dir, output_json_dir)
