# SLAM-Based Map Recovery from Dash Cam Videos

Purpose: This project aims to `recover a map` from a video recorded using a dash cam mounted on a moving car. The goal is to implement a `feature-based SLAM` system that tracks features across consecutive video frames to reconstruct a map of the environment.

## Approach
I employed a feature-based SLAM approach, leveraging OpenCV's powerful image processing and computer vision capabilities. The primary steps involved in the project are:

1. *Video Processing and Display*:
    - Using OpenCV (`cv2`) to read and display video frames for visualization.
    - Processing each frame to extract relevant features for tracking.

2. *Feature Extraction*:
    - Due to the presence of many trees in the video, which provide good nature features, I focus on tracking frame-to-frame features efficiently.
    - ORB (Oriented FAST and Rotated BRIEF) is used to identify keypoints and compute their descriptors in each frame.
    - The default ORB keypoint distribution is often poor (clustering in certain areas), leading to a need for a custom feature extraction strategy.


