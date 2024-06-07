from imutils import paths
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load YOLOv4-tiny model and classes
net = cv2.dnn.readNetFromDarknet('./weights-and-names/yolov4-tiny.cfg', './weights-and-names/yolov4-tiny.weights')
classes = []
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

def find_marker(image, confidence_threshold=0.2):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    frame_height, frame_width = image.shape[:2]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "basketball":
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                return (x, y, w, h)
    return None

KNOWN_DISTANCE = 65/12  # Preset known distance from the camera to the object
KNOWN_WIDTH = 9.5/12  # Convert inches to feet

# Initialize lists to store displacement values
horizontal_displacements = []
vertical_displacements = []

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

center_x_pixel = center_y_pixel = None

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break
    if center_x_pixel is None:
        center_x_pixel = image.shape[1] // 2  # Horizontal center
    if center_y_pixel is None:
        center_y_pixel = image.shape[0] // 2  # Vertical center

    marker = find_marker(image)
    if marker:
        x, y, w, h = marker
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        horizontal_displacement = (x + w / 2 - center_x_pixel) * (KNOWN_WIDTH / w)
        vertical_displacement = (y + h / 2 - center_y_pixel) * (KNOWN_WIDTH / h)

        # Store the displacement values
        if len(horizontal_displacements) < 60:
            horizontal_displacements.append(horizontal_displacement)
            vertical_displacements.append(vertical_displacement)

        cv2.putText(image, f"Horiz Disp: {horizontal_displacement:.2f} ft",
                    (50, image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)  # Bold red
        cv2.putText(image, f"Vert Disp: {vertical_displacement:.2f} ft",
                    (50, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Live Feed", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Stop capturing frames if we have n displacement values
    if len(horizontal_displacements) >= 60:
        break

cap.release()
cv2.destroyAllWindows()

# Plot the displacement values
plt.figure(figsize=(10, 10))

# Plot horizontal displacement
plt.subplot(2, 1, 1)
plt.plot(horizontal_displacements, range(len(horizontal_displacements)), 'ro-')
plt.title('Horizontal Displacement')
plt.ylabel('Frame Index')
plt.xlabel('Displacement (ft)')

# Plot vertical displacement
plt.subplot(2, 1, 2)
plt.plot(vertical_displacements, range(len(vertical_displacements)), 'go-')
plt.title('Vertical Displacement')
plt.ylabel('Frame Index')
plt.xlabel('Displacement (ft)')

plt.tight_layout()
plt.show()

#&---------------------------------------------------------------------------------------

# from imutils import paths
# import numpy as np
# import cv2

# # Load the model and classes
# net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
# classes = []
# with open("coco.names", "r") as f:
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
#             if confidence > 0.5 and classes[class_id] == "basketball":
#                 center_x = int(detection[0] * frame_width)
#                 center_y = int(detection[1] * frame_height)
#                 w = int(detection[2] * frame_width)
#                 h = int(detection[3] * frame_height)
#                 return ((center_x, center_y), (w, h), 0)
#     return None

# KNOWN_WIDTH = 9.5 / 12  # Convert inches to feet for the known width of the object
# KNOWN_DISTANCE = 47.5  # Preset known distance

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# center_x_pixel = center_y_pixel = None

# while True:
#     ret, image = cap.read()
#     if not ret:
#         break

#     if center_x_pixel is None:
#         center_x_pixel = image.shape[1] // 2  # Horizontal center
#     if center_y_pixel is None:
#         center_y_pixel = image.shape[0] // 2  # Vertical center

#     marker = find_marker(image)
#     if marker:
#         ((center_x, center_y), width_in_pixels) = marker
#         scaling_factor = KNOWN_WIDTH / width_in_pixels

#         horizontal_displacement = (center_x - center_x_pixel) * scaling_factor
#         vertical_displacement = (center_y - center_y_pixel) * scaling_factor

#         # Draw the bounding box around the detected object
#         box = cv2.boxPoints(marker)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

#         # Display displacements
#         cv2.putText(image, f"Horiz Disp: {horizontal_displacement:.2f} ft",
#                     (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
#         cv2.putText(image, f"Vert Disp: {vertical_displacement:.2f} ft",
#                     (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 69, 255), 3)

#         cv2.imshow("Live Feed", image)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import numpy as np
# import cv2

# def estimate_distance(knownWidth, focalLength, perWidth):
#     # Estimate the distance based on the object width in pixels and known width
#     return (knownWidth * focalLength) / perWidth

# def find_marker(image):
#     blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     frame_width = image.shape[1]
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and classes[class_id] == "basketball":
#                 center_x = int(detection[0] * frame_width)
#                 center_y = int(detection[1] * frame_height)
#                 w = int(detection[2] * frame_width)
#                 return ((center_x, center_y), w)  # Return only necessary data
#     return None

# # Main Loop
# cap = cv2.VideoCapture(0)
# KNOWN_WIDTH = 13.78 / 12  # inches to feet, adjust as necessary
# focalLength = 300  # Adjust based on your calibration results

# while True:
#     ret, image = cap.read()
#     if not ret:
#         break
#     marker = find_marker(image)
#     if marker:
#         center_x_pixel = image.shape[1] // 2
#         center_y_pixel = image.shape[0] // 2
#         ((center_x, center_y), width_in_pixels) = marker

#         distance = estimate_distance(KNOWN_WIDTH, focalLength, width_in_pixels)
#         scaling_factor = KNOWN_WIDTH / width_in_pixels

#         horizontal_displacement = (center_x - center_x_pixel) * scaling_factor
#         vertical_displacement = (center_y - center_y_pixel) * scaling_factor

#         # Drawing and displaying info
#         cv2.putText(image, f"Distance: {distance:.2f} ft", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(image, f"Horiz Disp: {horizontal_displacement:.2f} ft", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(image, f"Vert Disp: {vertical_displacement:.2f} ft", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 69, 255), 2)
        
#         cv2.imshow("Live Feed", image)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()






#remove distance from camera, only have horiz and vert displacement
#combat large fluctuations in displacement - start farther away
# from imutils import paths
# import numpy as np
# import imutils
# import cv2

# net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
# classes = []
# with open("coco.names", "r") as f:
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
#             if confidence > 0.5 and classes[class_id] == "basketball":
#                 center_x = int(detection[0] * frame_width)
#                 center_y = int(detection[1] * frame_height)
#                 w = int(detection[2] * frame_width)
#                 h = int(detection[3] * frame_height)
#                 return ((center_x, center_y), (w, h), 0)
#     return None

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# # Initialize center of frame
# center_x_pixel = None
# center_y_pixel = None

# while True:
#     ret, image = cap.read()
#     if not ret:
#         break
#     marker = find_marker(image)
#     if marker:
#         if center_x_pixel is None:
#             center_x_pixel = image.shape[1] // 2  # Find horizontal center once
#         if center_y_pixel is None:
#             center_y_pixel = image.shape[0] // 2  # Find vertical center once

#         # Calculating displacements
#         horizontal_displacement = (marker[0][0] - center_x_pixel) * (2.8 / marker[1][0])
#         vertical_displacement = (marker[0][1] - center_y_pixel) * (2.8 / marker[1][0])

#         box = cv2.boxPoints(marker)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

#         cv2.putText(image, f"Horiz Disp: {horizontal_displacement:.2f} ft",
#                     (image.shape[1] - 700, image.shape[0] - 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

#         cv2.putText(image, f"Vert Disp: {vertical_displacement:.2f} ft",
#                     (image.shape[1] - 700, image.shape[0] - 120),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 69, 255), 3)

#         cv2.imshow("Live Feed", image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # #&distance from camera, horizontal and vertical displacement

# from imutils import paths
# import numpy as np
# import imutils
# import cv2

# net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
# layer_names = net.getLayerNames()
# output_layers_indices = net.getUnconnectedOutLayers()
# output_layers = []
# for i in output_layers_indices:
#     if isinstance(i, np.ndarray):
#         output_layers.append(layer_names[i[0] - 1])
#     else:
#         output_layers.append(layer_names[i - 1])

# # def find_marker(image):
# #     blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
# #     net.setInput(blob)
# #     outs = net.forward(output_layers)
# #     frame_height, frame_width, channels = image.shape
# #     for out in outs:
# #         for detection in out:
# #             scores = detection[5:]
# #             class_id = np.argmax(scores)
# #             confidence = scores[class_id]
# #             if confidence > 0.5 and classes[class_id] == "cell phone":
# #                 center_x = int(detection[0] * frame_width)
# #                 center_y = int(detection[1] * frame_height)
# #                 w = int(detection[2] * frame_width)
# #                 h = int(detection[3] * frame_height)
# #                 return ((center_x, center_y), (w, h), 0)  
# #     return None

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
#                 return ((center_x, center_y), (w, h), 0)
#     return None

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     return (knownWidth * focalLength) / perWidth

# KNOWN_DISTANCE = 11.0 
# KNOWN_WIDTH = 2.8 / 12  #inch to feet 

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# #focal length initialization
# focalLength = None
# center_x_pixel = None
# center_y_pixel = None

# while focalLength is None:
#     ret, image = cap.read()
#     if not ret:
#         print("Failed to grab frame, retrying...")
#         continue
#     if center_x_pixel is None:
#         center_x_pixel = image.shape[1] // 2  #horiz center
#     if center_y_pixel is None:
#         center_y_pixel = image.shape[0] // 2  #vertical center
#     marker = find_marker(image)
#     if marker:
#         focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# while True:
#     ret, image = cap.read()
#     if not ret:
#         break
#     marker = find_marker(image)
#     if marker and focalLength is not None:
#         inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
#         horizontal_scaling_factor = KNOWN_WIDTH / marker[1][0]
#         horizontal_displacement = (marker[0][0] - center_x_pixel) * horizontal_scaling_factor
#         vertical_displacement = (marker[0][1] - center_y_pixel) * horizontal_scaling_factor 

#         box = cv2.boxPoints(marker)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
#         cv2.putText(image, f"Distance: {inches / 12:.2f} ft",
#             (image.shape[1] - 700, image.shape[0] - 20),
#             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

#         cv2.putText(image, f"Horiz Disp: {horizontal_displacement:.2f} ft",
#             (image.shape[1] - 700, image.shape[0] - 70),  
#             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

#         cv2.putText(image, f"Vert Disp: {vertical_displacement:.2f} ft",
#             (image.shape[1] - 700, image.shape[0] - 120),  
#             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 69, 255), 3)

#         cv2.imshow("Live Feed", image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


#prev attempts


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
# 	# compute and return the distance from the maker to the camera
# 	return (knownWidth * focalLength) / perWidth


# # initialize the known distance from the camera to the object, which
# # in this case is 11 inches
# KNOWN_DISTANCE = 11.0
# # initialize the known object width, which in this case, the piece of
# # paper is 2.8 inches wide (phone)
# KNOWN_WIDTH = 2.8
# # load the furst image that contains an object that is KNOWN TO BE 2 feet
# # from our camera, then find the paper marker in the image, and initialize
# # the focal length
# image = cv2.imread("images/11inches.jpg")
# marker = find_marker(image)
# focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# # loop over the images
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


# from imutils import paths
# import numpy as np
# import imutils
# import cv2

# def find_marker(image):
#     # convert the image to grayscale, blur it, and detect edges
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(gray, 35, 125)
#     # find the contours in the edged image and keep the largest one;
#     # we'll assume that this is our piece of paper in the image
#     cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     c = max(cnts, key=cv2.contourArea)
#     # compute the bounding box of the of the paper region and return it
#     return cv2.minAreaRect(c)

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     # compute and return the distance from the maker to the camera
#     return (knownWidth * focalLength) / perWidth

# # initialize the known distance from the camera to the object, which
# # in this case is 11 inches
# KNOWN_DISTANCE = 11.0
# # initialize the known object width, which in this case, the piece of
# # paper is 2.8 inches wide (phone)
# KNOWN_WIDTH = 2.8
# # start video capture
# cap = cv2.VideoCapture(0)

# # use the first frame to establish the focal length
# ret, image = cap.read()
# if not ret:
#     print("Failed to grab frame")
#     cap.release()
#     exit()

# marker = find_marker(image)
# focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# # Process frames from the video stream
# while True:
#     ret, image = cap.read()
#     if not ret:
#         break

#     marker = find_marker(image)
#     if marker:  # Check if a marker was found
#         inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
#         box = cv2.boxPoints(marker)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
#         cv2.putText(image, "%.2fft" % (inches / 12),
#                     (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
#                     2.0, (0, 255, 0), 3)
#     cv2.imshow("image", image)
#     if cv2.waitKey(1) == ord('q'):  # Exit if 'q' is pressed
#         break

# cap.release()
# cv2.destroyAllWindows()



# from imutils import paths
# import numpy as np
# import imutils
# import cv2

# net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
# classes = []
# with open("coco.names", "r") as f:
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
# image = cv2.imread("images/11inches.jpg")
# marker = find_marker(image)
# if marker:
#     focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
# else:
#     focalLength = None

# # loop over the images
# for imagePath in sorted(paths.list_images("images")):
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


#&distance from camera code

# from imutils import paths
# import numpy as np
# import imutils
# import cv2

# # Load YOLO
# net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
# classes = []
# with open("coco.names", "r") as f:
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
#                 return ((center_x, center_y), (w, h), 0)  # Dummy angle
#     return None

# def distance_to_camera(knownWidth, focalLength, perWidth):
#     return (knownWidth * focalLength) / perWidth

# KNOWN_DISTANCE = 11.0  # Set this based on your actual known distance
# KNOWN_WIDTH = 2.8      # Set this based on your actual object's width

# # Set up video capture
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# # Use the first frame to establish the focal length
# ret, image = cap.read()
# if not ret:
#     print("Failed to grab frame")
#     cap.release()
#     exit()

# focalLength = None
# attempts = 0
# while focalLength is None:
#     ret, image = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         attempts += 1
#         if attempts >= 30:  # Limit attempts to 10
#             print("Failed to initialize after several attempts.")
#             cap.release()
#             exit()
#         continue

#     marker = find_marker(image)
#     if marker:
#         focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
#     else:
#         print("Trying to initialize focal length...")

# # Main loop for the live feed
# while True:
#     ret, image = cap.read()
#     if not ret:
#         break

#     marker = find_marker(image)
#     if marker and focalLength is not None:
#         inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
#         box = cv2.boxPoints(marker)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
#         cv2.putText(image, "%.2fft" % (inches / 12),
#                     (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
#                     2.0, (0, 255, 0), 3)

#     cv2.imshow("Live Feed", image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#         break

# cap.release()
# cv2.destroyAllWindows()






