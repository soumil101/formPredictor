import os
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog

# Create a directory to save frames if it does not exist
frames_dir = 'frames'
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the root window

# Prompt the user for a video file to upload using file explorer
video_path = filedialog.askopenfilename(title="Select a video file",
                                        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])

if not video_path:
    print("No file selected. Exiting.")
    exit(1)

# Ask the user how many frames they want to split the video into
num_frames = simpledialog.askinteger("Input", "Enter the number of frames you want the video to be split into:")

if not num_frames:
    print("No number of frames entered. Exiting.")
    exit(1)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the specific frames to save
frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

# Initialize a counter for the frames
frame_counter = 0
saved_frame_counter = 0

while True:
    success, frame = cap.read()
    
    if not success:
        break

    if frame_counter in frame_indices:
        frame_filename = os.path.join(frames_dir, f'frame_{saved_frame_counter}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_frame_counter += 1
    
    frame_counter += 1

# Release the video capture object
cap.release()

print(f"Video split into {saved_frame_counter} frames and saved in the '{frames_dir}' directory.")
