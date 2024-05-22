import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
from ball_tracking import track_ball, print_ball_data
import av

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

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return av.VideoFrame.from_ndarray(self.frame, format="bgr24")

    def process(self):
        if self.frame is None:
            return

        frame = self.frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
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

        for i in indices:
            box = boxes[i]
            class_id = class_ids[i]
            color = [int(c) for c in colors[class_id]]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), color, 2)
            text = f"{classes[class_id]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def main():
    st.title("Live Video Processing with Streamlit")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.process()

if __name__ == "__main__":
    main()
