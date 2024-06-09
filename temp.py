import streamlit as st
import cv2
import tempfile
import os

# File uploader widget
uploaded_file = st.file_uploader("Choose a .mov file", type=["mov"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mov") as tmp_file:
        # Write the uploaded file's content to the temporary file
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name
    
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(temp_file_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
    else:
        while cap.isOpened():
            # Read frame-by-frame
            ret, frame = cap.read()
            
            # If frame is read correctly ret is True
            if not ret:
                st.write("End of video.")
                break
            
            # Display the frame (converted to RGB for Streamlit compatibility)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb)
            
            # Break the loop on user interruption
            if st.button("Stop"):
                break

        # When everything done, release the video capture object
        cap.release()

    # Clean up the temporary file
    os.remove(temp_file_path)
