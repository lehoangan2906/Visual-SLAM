#!/usr/bin/python3

import cv2 
import pygame

# Create an ORB object
orb = cv2.ORB_create()


def process_frame(img):
    height, width= img.shape[:2]
    img = cv2.resize(img, (width//4, height//4)) # Downscale the image

    # Detect the keypoints and descriptors in the frame with ORB
    kp1, des1 = orb.detectAndCompute(img, None)

    # Display the resulting frame
    cv2.imshow("frame", img)
    
    # Need to have to press a random key to move from frame to frame
    # If you want to play the video in real time, change the waitkey value to 1
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



