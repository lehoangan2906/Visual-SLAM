#!/usr/bin/python3

import cv2 
import numpy as np
from Extractor.extractor import Extractor

# Call an instance of the Extractor class
# Extract keypoints and descriptors from each frame, then match them all.
fe = Extractor()

def process_frame(img):
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4)) # Downscale the image

    matches = fe.extract(img) # Extract keypoints and descriptors from the frame

    if matches is None:
        return
    
    # Retrieve the keypoints from last frame 
    kps = fe.last["kps"] if fe.last is not None else []

    # Draw the keypoints on the frame
    for p in kps:
        x, y = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (x, y), color=(0, 255, 0), radius=3) # Draw green keypoints

    # Draw matches between the current and previous frame
    if matches:
        for p1, p2 in matches:
            x1, y1 = map(lambda coord: int(round(coord)), p1.pt)
            x2, y2 = map(lambda coord: int(round(coord)), p2.pt)
            cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0)) # Draw blue lines between the keypoints

    # Display the processed frame
    cv2.imshow("frame", img)
    cv2.waitKey(1)

if __name__ == "__main__":

    # Create a video capture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name.
    cap = cv2.VideoCapture("videos/test.mp4")

    # Read until video is completed
    while cap.isOpened:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        process_frame(frame)