# import numpy as np
# import cv2
# from imutils import paths

# def load_yolo():
#     net = cv2.dnn.readNet("./weights-and-names/yolov4-tiny.weights", "./weights-and-names/yolov4-tiny.cfg")
#     with open("./weights-and-names/coco.names", "r") as f:
#         classes = [line.strip() for line in f.readlines()]
#     layer_names = net.getLayerNames()
#     output_layers_indices = net.getUnconnectedOutLayers()
#     output_layers = []
#     for i in output_layers_indices:
#         if isinstance(i, np.ndarray):
#             output_layers.append(layer_names[i[0] - 1])
#         else:
#             output_layers.append(layer_names[i - 1])
#     return net, classes, output_layers

# def detect_objects(img, net, output_layers):
#     height, width, channels = img.shape
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     class_ids, confidences, boxes = [], [], []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Confidence threshold
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     return boxes, confidences, class_ids, indexes

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     return (knownWidth * focalLength) / perWidth

# # Load YOLO model
# net, classes, output_layers = load_yolo()

# # Initialize known distance and width
# KNOWN_DISTANCE = 24.0
# KNOWN_WIDTH = 2.8  # Average width of a cell phone in inches

# # Load the first image to calculate focal length
# base_image_path = "./images/11inches.jpg"  # Update this path as necessary
# image = cv2.imread(base_image_path)
# boxes, confidences, class_ids, indexes = detect_objects(image, net, output_layers)
# for i in range(len(boxes)):
#     if i in indexes and classes[class_ids[i]] == "cell phone":
#         x, y, w, h = boxes[i]
#         focalLength = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
#         break

# # Loop over images in the frames directory
# for imagePath in sorted(paths.list_images("./images")):
#     image = cv2.imread(imagePath)
#     if image is None:
#         print(f"Image not found or unable to load: {imagePath}")
#         continue

#     boxes, confidences, class_ids, indexes = detect_objects(image, net, output_layers)
#     cell_phone_detected = False
#     for i in range(len(boxes)):
#         if i in indexes and classes[class_ids[i]] == "cell phone":
#             x, y, w, h = boxes[i]
#             inches = distance_to_camera(KNOWN_WIDTH, focalLength, w)
#             print(f"Detected cell phone in image {imagePath} with bounding box: {x, y, w, h}")
#             print(f"Calculated distance: {inches} inches")
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(image, "%.2fft" % (inches / 12), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             cell_phone_detected = True
#     if not cell_phone_detected:
#         print(f"No cell phone detected in image {imagePath}.")
#     cv2.imshow("image", image)
#     cv2.waitKey(0)

#& attempt 2

import numpy as np
import cv2
from imutils import paths

def load_yolo():
    net = cv2.dnn.readNet("./weights-and-names/yolov4-tiny.weights", "./weights-and-names/yolov4-tiny.cfg")
    with open("./weights-and-names/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = []
    for i in output_layers_indices:
        if isinstance(i, np.ndarray):
            output_layers.append(layer_names[i[0] - 1])
        else:
            output_layers.append(layer_names[i - 1])
    return net, classes, output_layers

def detect_objects(img, net, output_layers, confidence_threshold=0.2):  # Lowered confidence threshold
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    return boxes, confidences, class_ids, indexes

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

# Load YOLO model
net, classes, output_layers = load_yolo()

# Print loaded classes to ensure "cell phone" is included
print("Loaded classes: ", classes)

# Initialize known distance and width
KNOWN_DISTANCE = 24.0
KNOWN_WIDTH = 2.8  # Average width of a cell phone in inches

# Load the first image to calculate focal length
base_image_path = "./frames/frame_0.jpg"  # Update this path as necessary
image = cv2.imread(base_image_path)
if image is None:
    raise ValueError(f"Base image not found at path: {base_image_path}")

boxes, confidences, class_ids, indexes = detect_objects(image, net, output_layers)
focalLength = None

# Draw all detected boxes for debugging
for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if i in indexes and classes[class_ids[i]] == "cell phone":
        focalLength = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
        print(f"Detected cell phone in base image with bounding box: {x, y, w, h}")
        print(f"Calculated focal length: {focalLength}")

cv2.imshow("Base Image Detections", image)
cv2.waitKey(0)

if focalLength is None:
    raise ValueError("No cell phone detected in the base image to calculate focal length.")

# Loop over images in the frames directory
for imagePath in sorted(paths.list_images("./frames")):
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Image not found or unable to load: {imagePath}")
        continue

    boxes, confidences, class_ids, indexes = detect_objects(image, net, output_layers)
    cell_phone_detected = False
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw all detections
        if i in indexes and classes[class_ids[i]] == "cell phone":
            inches = distance_to_camera(KNOWN_WIDTH, focalLength, w)
            print(f"Detected cell phone in image {imagePath} with bounding box: {x, y, w, h}")
            print(f"Calculated distance: {inches} inches")
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "%.2fft" % (inches / 12), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cell_phone_detected = True
    if not cell_phone_detected:
        print(f"No cell phone detected in image {imagePath}.")
    cv2.imshow("image", image)
    cv2.waitKey(0)

#& attempt 3

# from imutils import paths
# import numpy as np
# import imutils
# import cv2

# net = cv2.dnn.readNetFromDarknet('./weights-and-names/yolov4.cfg', './weights-and-names/yolov4.weights')
# classes = []
# with open("./weights-and-names/coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
# layer_names = net.getLayerNames()
# output_layers_indices = net.getUnconnectedOutLayers()
# output_layers = []
# for i in output_layers_indices:
#     if isinstance(i, np.ndarray):
#         output_layers.append(layer_names[i[0] - 1])
#     else:
#         output_layers.append(layer_names[i - 1])

# def find_marker(image):
#     blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     frame_height, frame_width, channels = image.shape

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and classes[class_id] == "cell phone":
#                 center_x = int(detection[0] * frame_width)
#                 center_y = int(detection[1] * frame_height)
#                 w = int(detection[2] * frame_width)
#                 h = int(detection[3] * frame_height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 return ((center_x, center_y), (w, h), 0)  # Dummy angle
#     return None

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     # compute and return the distance from the maker to the camera
#     return (knownWidth * focalLength) / perWidth

# # initialize the known distance from the camera to the object, which
# # in this case is 11 inches
# KNOWN_DISTANCE = 11.0
# # initialize the known object width, which in this case, the piece of
# # paper is 2.8 inches wide (phone)
# KNOWN_WIDTH = 2.8
# # load the furst image that contains an object that is KNOWN TO BE 2 feet
# # from our camera, then find the paper marker in the image, and initialize
# # the focal length
# image = cv2.imread("frames/frame_0.jpg")
# marker = find_marker(image)
# if marker:
#     focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
# else:
#     focalLength = None

# # loop over the images
# for imagePath in sorted(paths.list_images("frames")):
#     image = cv2.imread(imagePath)
#     marker = find_marker(image)
#     if marker and focalLength is not None:
#         inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
#         box = cv2.boxPoints(marker)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
#         cv2.putText(image, "%.2fft" % (inches / 12),
#                     (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
#                     2.0, (0, 255, 0), 3)
#     cv2.imshow("image", image)
#     # Use '0' to proceed to the next image
#     key = cv2.waitKey(0)
#     if key == ord('q') or key == 27:  # ASCII value for ESC
#         break
#     elif key != ord('0'):
#         continue

# cv2.destroyAllWindows()

#& attempt 4


# import numpy as np
# import cv2
# from imutils import paths
# import os

# def load_yolo():
#     weight_path = "./weights-and-names/yolov4-tiny.weights"
#     cfg_path = "./weights-and-names/yolov4-tiny.cfg"
#     names_path = "./weights-and-names/coco.names"

#     # Check if files exist
#     if not os.path.isfile(weight_path):
#         raise ValueError(f"Weight file not found: {weight_path}")
#     if not os.path.isfile(cfg_path):
#         raise ValueError(f"Config file not found: {cfg_path}")
#     if not os.path.isfile(names_path):
#         raise ValueError(f"Names file not found: {names_path}")

#     net = cv2.dnn.readNet(weight_path, cfg_path)
#     with open(names_path, "r") as f:
#         classes = [line.strip() for line in f.readlines()]
#     layer_names = net.getLayerNames()
#     output_layers_indices = net.getUnconnectedOutLayers()
#     output_layers = []
#     for i in output_layers_indices:
#         if isinstance(i, np.ndarray):
#             output_layers.append(layer_names[i[0] - 1])
#         else:
#             output_layers.append(layer_names[i - 1])
#     return net, classes, output_layers

# def detect_objects(img, net, output_layers, confidence_threshold=0.2):  # Lowered confidence threshold
#     height, width, channels = img.shape
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     class_ids, confidences, boxes = [], [], []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > confidence_threshold:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
#     return boxes, confidences, class_ids, indexes

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     return (knownWidth * focalLength) / perWidth

# # Load YOLO model
# net, classes, output_layers = load_yolo()

# # Print loaded classes to ensure "cell phone" is included
# print("Loaded classes: ", classes)

# # Initialize known distance and width
# KNOWN_DISTANCE = 24.0
# KNOWN_WIDTH = 2.8  # Average width of a cell phone in inches

# # Load the base image to calculate focal length
# base_image_path = "./images/11inches.jpg"  # Path to the uploaded image
# image = cv2.imread(base_image_path)
# if image is None:
#     raise ValueError(f"Base image not found at path: {base_image_path}")

# boxes, confidences, class_ids, indexes = detect_objects(image, net, output_layers)
# focalLength = None

# # Draw all detected boxes for debugging
# for i in range(len(boxes)):
#     x, y, w, h = boxes[i]
#     label = classes[class_ids[i]]
#     confidence = confidences[i]
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     if i in indexes and classes[class_ids[i]] == "cell phone":
#         focalLength = (w * KNOWN_DISTANCE) / KNOWN_WIDTH
#         print(f"Detected cell phone in base image with bounding box: {x, y, w, h}")
#         print(f"Calculated focal length: {focalLength}")

# cv2.imshow("Base Image Detections", image)
# cv2.waitKey(0)

# if focalLength is None:
#     raise ValueError("No cell phone detected in the base image to calculate focal length.")

# # Loop over images in the frames directory
# for imagePath in sorted(paths.list_images("./images")):
#     image = cv2.imread(imagePath)
#     if image is None:
#         print(f"Image not found or unable to load: {imagePath}")
#         continue

#     boxes, confidences, class_ids, indexes = detect_objects(image, net, output_layers)
#     cell_phone_detected = False
#     for i in range(len(boxes)):
#         x, y, w, h = boxes[i]
#         label = classes[class_ids[i]]
#         confidence = confidences[i]
#         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw all detections
#         cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         if i in indexes and classes[class_ids[i]] == "cell phone":
#             inches = distance_to_camera(KNOWN_WIDTH, focalLength, w)
#             print(f"Detected cell phone in image {imagePath} with bounding box: {x, y, w, h}")
#             print(f"Calculated distance: {inches} inches")
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(image, "%.2fft" % (inches / 12), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             cell_phone_detected = True
#     if not cell_phone_detected:
#         print(f"No cell phone detected in image {imagePath}.")
#     cv2.imshow("image", image)
#     cv2.waitKey(0)
