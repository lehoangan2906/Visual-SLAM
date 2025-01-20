#!/usr/bin/python3

import cv2 
import numpy as np

class FeatureExtractor(object):
    GX = 8
    GY = 6

    def __init__(self):
        self.orb = cv2.ORB_create()

    def extract(self, img):
        # Use OpenCV goodFeaturesToTrack to detect strong corners (keypoints)
        feats = cv2.goodFeaturesToTrack(
            np.mean(img, axis=2).astype(np.uint8), # Convert the image to grayscale
            maxCorners=3000,                       # Maximum number of corners to return
            qualityLevel=0.01,                     # Minimum quality of corners
            minDistance=3                          # Minimum Euclidean distance between corners
        )

        # Convert detected corners to KeyPoint objects
        kps = [cv2.KeyPoint(x=f[0][0], y = f[0][1], size=20) for f in feats]    # Extract the x and y coordinates of the corner then assign a fixed size of 20 pixels for each keypoint.

        # Compute the keypoint descriptor with ORB
        des = self.orb.compute(img, kps)
        return kps, des


fe = FeatureExtractor()

def process_frame(img):
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4)) # Downscale the image

    kps, des = fe.extract(img)

    # Draw the keypoints on the frame
    for p in kps:
        x, y = map(lambda x: int(round(x)), p.pt)
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