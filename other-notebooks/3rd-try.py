#^ original

# # import the necessary packages
# from imutils import paths
# import numpy as np
# import imutils
# import cv2
# def find_marker(image):
# 	# convert the image to grayscale, blur it, and detect edges
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	gray = cv2.GaussianBlur(gray, (5, 5), 0)
# 	edged = cv2.Canny(gray, 35, 125)
# 	# find the contours in the edged image and keep the largest one;
# 	# we'll assume that this is our piece of paper in the image
# 	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 	cnts = imutils.grab_contours(cnts)
# 	c = max(cnts, key = cv2.contourArea)
# 	# compute the bounding box of the of the paper region and return it
# 	return cv2.minAreaRect(c)

# def distance_to_camera(knownWidth, focalLength, perWidth):
# # compute and return the distance from the maker to the camera
#     return (knownWidth * focalLength) / perWidth

# # initialize the known distance from the camera to the object, which
# # in this case is 24 inches
# KNOWN_DISTANCE = 24.0
# # initialize the known object width, which in this case, the piece of
# # paper is 12 inches wide
# KNOWN_WIDTH = 11.0
# # load the furst image that contains an object that is KNOWN TO BE 2 feet
# # from our camera, then find the paper marker in the image, and initialize
# # the focal length
# image = cv2.imread("images/2ft.png")
# marker = find_marker(image)
# focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# for imagePath in sorted(paths.list_images("images")):
# 	# load the image, find the marker in the image, then compute the
# 	# distance to the marker from the camera
# 	image = cv2.imread(imagePath)
# 	marker = find_marker(image)
# 	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
# 	# draw a bounding box around the image and display it
# 	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
# 	box = np.int0(box)
# 	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
# 	cv2.putText(image, "%.2fft" % (inches / 12),
# 		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
# 		2.0, (0, 255, 0), 3)
# 	cv2.imshow("image", image)
# 	cv2.waitKey(0)

#^v2

# import the necessary packages
# import the necessary packages
# from imutils import paths
# import numpy as np
# import cv2

# # Load YOLO
# net = cv2.dnn.readNet("weights-and-names/yolov4-tiny.weights", "weights-and-names/yolov4-tiny.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# classes = []
# with open("weights-and-names/coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Find the class ID for "cell phone"
# cell_phone_class_id = classes.index("book")

# # Function to find the cell phone using YOLO
# def find_marker(image):
#     height, width = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Initialization
#     confidences = []
#     boxes = []

#     # For each detection from each output layer
#     # get the confidence, class id, bounding box params and ignore weak detections
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             # Only proceed if the detected object is a cell phone
#             if class_id == cell_phone_class_id and confidence > 0.5:  # You can change this threshold to suit your needs
#                 # Cell phone detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))

#     # Apply non-max suppression to remove overlapping bounding boxes
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     if len(indexes) > 0:
#         # Assume the first detected object is the desired one
#         i = indexes[0]
#         box = boxes[i]
#         x, y, w, h = box[0], box[1], box[2], box[3]
#         return (x, y, w, h)
#     return None

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     # compute and return the distance from the marker to the camera
#     return (knownWidth * focalLength) / perWidth

# # initialize the known distance from the camera to the object
# KNOWN_DISTANCE = 24.0  # example: 24 inches
# # initialize the known object width
# KNOWN_WIDTH = 3.0  # example: 3 inches (update with the actual width of your cell phone or basketball)

# # load the base image that contains the object at a known distance
# base_image_path = "frames/frame_0.jpg"  # path to the base image
# image = cv2.imread(base_image_path)
# marker = find_marker(image)
# if marker is None:
#     print("Marker not found in base image.")
# else:
#     focalLength = (marker[2] * KNOWN_DISTANCE) / KNOWN_WIDTH

#     # iterate over frames in the frames directory
#     for imagePath in sorted(paths.list_images("frames")):
#         if "frame_0.jpg" in imagePath:
#             continue  # skip the base image
        
#         # load the image, find the marker in the image, then compute the distance to the marker from the camera
#         image = cv2.imread(imagePath)
#         marker = find_marker(image)
#         if marker is None:
#             print(f"Marker not found in image: {imagePath}")
#             continue
#         inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[2])
        
#         # draw a bounding box around the image and display it
#         x, y, w, h = marker
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(image, "%.2fft" % (inches / 12),
#                     (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
#                     2.0, (0, 255, 0), 3)
#         cv2.imshow("image", image)
#         cv2.waitKey(0)

#& 3rd try

# from imutils import paths
# import numpy as np
# import cv2

# # Load YOLOv4-tiny model and classes
# net = cv2.dnn.readNetFromDarknet('./weights-and-names/yolov4-tiny.cfg', './weights-and-names/yolov4-tiny.weights')
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
#     blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     frame_height, frame_width = image.shape[:2]
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
#                 return (x, y, w, h)
#     return None

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     # compute and return the distance from the marker to the camera
#     return (knownWidth * focalLength) / perWidth

# # Initialize known values
# KNOWN_DISTANCE = 24.0 / 12  # Preset known distance in feet
# KNOWN_WIDTH = 3.0 / 12  # Known width of the cell phone in feet

# # Load the base image that contains the object at a known distance
# base_image_path = "frames/frame_0.jpg"  # Path to the base image
# image = cv2.imread(base_image_path)
# marker = find_marker(image)
# if marker is None:
#     print("Marker not found in base image.")
#     exit()
# else:
#     focalLength = (marker[2] * KNOWN_DISTANCE) / KNOWN_WIDTH

# # Iterate over frames in the frames directory
# for imagePath in sorted(paths.list_images("frames")):
#     if "frame_0.jpg" in imagePath:
#         continue  # Skip the base image

#     # Load the image, find the marker in the image, then compute the distance to the marker from the camera
#     image = cv2.imread(imagePath)
#     marker = find_marker(image)
#     if marker is None:
#         print(f"Marker not found in image: {imagePath}")
#         continue
    
#     # Calculate distance
#     inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[2])

#     # Draw a bounding box around the image
#     x, y, w, h = marker
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Calculate horizontal and vertical displacement
#     center_x_pixel = image.shape[1] // 2
#     center_y_pixel = image.shape[0] // 2
#     horizontal_displacement = (x + w / 2 - center_x_pixel) * (KNOWN_WIDTH / w)
#     vertical_displacement = (y + h / 2 - center_y_pixel) * (KNOWN_WIDTH / h)

#     # Display distance and displacement on the image
#     cv2.putText(image, f"Distance: {inches:.2f} ft", (50, image.shape[0] - 150),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
#     cv2.putText(image, f"Horiz Disp: {horizontal_displacement:.2f} ft", (50, image.shape[0] - 100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)  # Bold red
#     cv2.putText(image, f"Vert Disp: {vertical_displacement:.2f} ft", (50, image.shape[0] - 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

#     # Display the image
#     cv2.imshow("Frame", image)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

#& just track a cell phone

import cv2
import os

# Load the pre-trained model for object detection
net = cv2.dnn.readNetFromCaffe("./weights-and-names/MobileNetSSD_deploy.prototxt", "./weights-and-names/MobileNetSSD_deploy.caffemodel")

# Define the directory containing the frames
frames_dir = "frames"

# Loop over all files in the frames directory
for filename in os.listdir(frames_dir):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(frames_dir, filename)
        image = cv2.imread(image_path)

        # Prepare the image for detection
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # Pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence and class ID
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            # Filter out weak detections
            if confidence > 0.5:
                # Get the class label
                class_label = ""
                if class_id == 6:
                    class_label = "Cell Phone"

                # Compute the bounding box coordinates
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label
                label = f"{class_label}: {confidence * 100:.2f}%"
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Display the image with detections
        cv2.imshow("Cell Phone Detection", image)
        cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()

