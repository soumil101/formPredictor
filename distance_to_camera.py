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

# import numpy as np
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

#&distance from camera and horizontal displacement

from imutils import paths
import numpy as np
import imutils
import cv2

# Load the model and classes
net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = []
for i in output_layers_indices:
    if isinstance(i, np.ndarray):
        output_layers.append(layer_names[i[0] - 1])
    else:
        output_layers.append(layer_names[i - 1])

# def find_marker(image):
#     blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
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

def find_marker(image):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    frame_height, frame_width = image.shape[:2]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "cell phone":
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)
                return ((center_x, center_y), (w, h), 0)  # Return early once a phone is detected
    return None

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

KNOWN_DISTANCE = 11.0  # Example known distance
KNOWN_WIDTH = 2.8 / 12  # Convert width to feet

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize focal length and center of frame
focalLength = None
center_x_pixel = None
center_y_pixel = None

while focalLength is None:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame, retrying...")
        continue
    if center_x_pixel is None:
        center_x_pixel = image.shape[1] // 2  # Find the horizontal center of the frame
    if center_y_pixel is None:
        center_y_pixel = image.shape[0] // 2  # Find the vertical center of the frame
    marker = find_marker(image)
    if marker:
        focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# Main processing loop
while True:
    ret, image = cap.read()
    if not ret:
        break
    marker = find_marker(image)
    if marker and focalLength is not None:
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
        horizontal_scaling_factor = KNOWN_WIDTH / marker[1][0]
        horizontal_displacement = (marker[0][0] - center_x_pixel) * horizontal_scaling_factor
        vertical_displacement = (marker[0][1] - center_y_pixel) * horizontal_scaling_factor  # Using the same scaling factor

        box = cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, f"Distance: {inches / 12:.2f} ft",
            (image.shape[1] - 700, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.putText(image, f"Horiz Disp: {horizontal_displacement:.2f} ft",
            (image.shape[1] - 700, image.shape[0] - 70),  # Increased spacing
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        cv2.putText(image, f"Vert Disp: {vertical_displacement:.2f} ft",
            (image.shape[1] - 700, image.shape[0] - 120),  # Further increased spacing
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 69, 255), 3)

        cv2.imshow("Live Feed", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
