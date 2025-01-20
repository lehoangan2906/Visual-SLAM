# SLAM-Based Map Recovery from Dash Cam Videos

Purpose: This project aims to `recover a map` from a video recorded using a dash cam mounted on a moving car. The goal is to implement a `feature-based SLAM` system that tracks features across consecutive video frames to reconstruct a map of the environment.

## Approach
I employed a feature-based SLAM approach, leveraging OpenCV's powerful image processing and computer vision capabilities. The primary steps involved in the project are:

1. **Video Processing and Display**:
    - Using OpenCV (`cv2`) to read and display video frames for visualization.
    - Processing each frame to extract relevant features for tracking.

2. **Feature Extraction**:
    - Due to the presence of many trees in the video, which provide good nature features, I focus on tracking frame-to-frame features efficiently.
    - **ORB** (Oriented FAST and Rotated BRIEF) is used to identify keypoints and compute their descriptors in each frame.
    - The default ORB keypoint distribution is often poor (clustering in certain areas), leading to a need for a custom feature extraction strategy.

3. **Custom ORB Feature Extraction**:
    - The frame is divided into a grid to ensure uniform distribution of keypoints.
    - ORB keypoints and descriptors are detected and computed within each grid cell.
    - The detected keypoints' coordinates are then shifted to match the global coordinate system of the full image.

4. **Keypoint Selection and Refinement**:
    - The newly detected keypoints often remain clustered affecting tracking accuracy.
    - To mitigate this, we utilize OpenCV's `cv2.goodFeaturestoTrack` to select high-quality corner features.
    - `goodFeaturesToTrack` only provides corner locations, but SLAM algorithms require robust **descriptors** (such as SIFT, ORB, or SURF) for effective feature matching between frames.

## Tools and Libraries Used

- OpenCV:

    - cv2.VideoCapture for video input

    - cv2.imshow for frame display

    - cv2.ORB_create for keypoint detection and description

    - cv2.goodFeaturesToTrack for selecting quality features

- NumPy:

    - Used for efficient matrix operations and handling image data

##Challenges and Solutions

1. Keypoint Distribution:

    - Challenge: Default ORB detection results in uneven keypoint distribution.

    - Solution: Implementing a grid-based approach to enforce uniformity.

2. Feature Clustering:

    - Challenge: Clustering of features even after grid-based detection.

    - Solution: Selecting the best features using goodFeaturesToTrack for improved distribution.

3. Feature Matching Across Frames:

    - Challenge: SLAM requires robust matching of features across multiple frames.

    - Solution: Using feature descriptors (ORB, SIFT, SURF) to ensure accurate matching.

##How to Run the Project
1. Install the required dependencies:
```bash
pip3 install opencv-python numpy
```
2. Run the SLAM pipeline with a video file:
```bash
chmod +x slam.py
./slam.py
```
3. Press `q` to exit the video display.
