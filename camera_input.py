import cv2
import numpy as np
from ball_tracking import track_ball, print_ball_data

pixel_width = 1920
pixel_height = 1080
width_in_feet = 50
height_in_feet = width_in_feet * (pixel_height / pixel_width) 

conversion_factor_x = width_in_feet / pixel_width
conversion_factor_y = height_in_feet / pixel_height

classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

conf_threshold = 0.5  
nms_threshold = 0.4   

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    outputs = net.forward(net.getUnconnectedOutLayersNames())

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    center_x_pixel = frame_width / 2
    center_y_pixel = frame_height / 2
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                x = int(detection[0] * frame_width)
                y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = x - width / 2
                top = y - height / 2
                adjusted_x = (left - center_x_pixel) * conversion_factor_x
                adjusted_y = (top - center_y_pixel) * conversion_factor_y
                adjusted_width = width * conversion_factor_x
                adjusted_height = height * conversion_factor_y
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([adjusted_x, adjusted_y, adjusted_width, adjusted_height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if isinstance(indices, tuple) and not indices:
        indices = []
    else:
        indices = indices.flatten()

    selected_boxes = [boxes[i] for i in indices]
    selected_class_ids = [class_ids[i] for i in indices]

    ball_data = track_ball(frame, selected_boxes, selected_class_ids, classes)
    print_ball_data(ball_data)

    cv2.imshow('Detected Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
