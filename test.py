#!/usr/bin/python3

import cv2 
import numpy as np

class FeatureExtractor(object):
    GX = 8
    GY = 6

    def __init__(self):
        self.orb = cv2.ORB_create()

    def extract(self, img):
        return cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), maxCorners=800, qualityLevel=0.01, minDistance=3)


fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4)) # Downscale the image

    kp = fe.extract(img)

    # Draw the keypoints on the frame
    for p in kp:
        x, y = map(lambda x: int(round(x)), p[0])
        cv2.circle(img, (x, y), color=(0, 255, 0), radius=3)

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