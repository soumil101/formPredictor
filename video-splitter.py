import os
import cv2
from tkinter import Tk, filedialog, simpledialog
import shutil

'''
upload the base image first (the calibration image for the known distance)

then upload the video
'''

def video_splitter(video_path, n_frames, output_dir="./frames"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < n_frames:
        n_frames = total_frames
        step = 1
    else:
        step = total_frames // (n_frames - 1)

    frame_num = 1  # Start from 1 since frame_0.jpg is the base image
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"frame_{frame_num}.jpg")
            cv2.imwrite(output_path, frame)
            frame_num += 1

    cap.release()

if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    # Prompt for base image first
    base_image_path = filedialog.askopenfilename(title="Select Base Image")
    if base_image_path:
        # Save base image as frame_0.jpg in the output directory
        output_dir = "./frames"
        os.makedirs(output_dir, exist_ok=True)
        base_image_dest = os.path.join(output_dir, "frame_0.jpg")
        shutil.copy(base_image_path, base_image_dest)
    else:
        print("No base image selected.")
        root.destroy()
        exit()

    # Prompt for video file
    video_path = filedialog.askopenfilename(title="Select Video File")
    if video_path:
        n_frames = simpledialog.askinteger("Number of Frames", "How many frames do you want the video split into?")
        if n_frames:
            # Process the video
            video_splitter(video_path, n_frames, output_dir)
        else:
            print("Invalid number of frames entered.")
    else:
        print("No video file selected.")

    root.destroy()
