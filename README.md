# Monocular Visual SLAM
SLAM is a technique used in robotics and computer vision to enable a device to build a map of an unknown environment while simultaneously determining its location within that map.

Monocular SLAM (which is built in this project), relies solely on a single camera, making it lightweight but challenging due to scale ambiguity and reliance on visual cues.

## Problem Statement
**Problem**: We have a video from a car’s dashcam → we want to draw a map of the surrounding environment (or at least the car’s moving trajectory). → we also want to localize the car within that map.

**Approach**: 

    - The video can be split into individual frames.
    - How do we get a map from these frames? → Look at how things move between frames.
    - How do we track movement? → Find keypoints (important spots) in each frame. → Match those keypoints across frames to see what moved where.
    - What do we do with matches? → Use the matches keypoints to estimate how the camera (car) moved. → Compute the camera’s rotation and position change (motion) using the matches.
    - How do we estimate motion → Calculate a matrix (like the essential matrix) that shows the camera’s shift. → Break it down into rotation and direction the car traveled.
    - What next for the map? → Use the motion and matched keypoints to find 3D locations of those points. → Triangulate by combining the camera’s positions to pinpoint where keypoints are in space.
    - Then what? → Build a 3D map from these points (a sparse cloud of landmarks). → Update the car’s position and the map with each new frame.

## Proposed Pipeline

```text
Pipeline: Twitch SLAM (Monocular SLAM Toy Implementation)

|__1. Input Acquisition and Preprocessing
|  |
|  |_1.1. Video Input 
|  |
|  |_1.2. Preprocessing 
|     |
|     |__1.2.1. Downscaling
|     |
|     |__1.2.2. Camera Calibration
|
|__2. Feature Detection and Matching
|  | 
|  |_2.1. Keypoint Detection
|  |
|  |_2.2. Feature Matching
|     |
|     |_2.2.1. Matching
|     |
|     |_2.2.2. Filtering
|
|__3. Camera Motion Estimation
|  |
|  |_3.1. Essential Matrix Computation
|  |
|  |_3.2. Pose Decomposition
|
|__4. Triangularation for 3D mapping
|  |
|  |_4.1. Projection Matrix formation
|  |
|  |_4.2. 3D Point Computation
|
|__5. Integration and Tracking
|  |
|  |_5.1. Pose Accumulation
|  |
|  |_5.2. Map Update
|
|__6. Visualization and Output
   |
   |_6.1. 2D Visualization
   |
   |_6.2. 3D Visualization
```
---
## Main Content

### 1. Input Acquisition and Preprocessing

- **Why It’s needed?** SLAM starts with raw sensory data  — here, a sequence of images from a monocular camera.  These images provide the visual information needed to detect landmarks and estimate motion. Preprocessing ensures the data is in a usable format (e.g., manageable resolution, grayscale for feature detection)
- **Method**:
    - Used OpenCV’s `cv2.VideoCapture` to read frames from a pre-recorded video file.
    - The script `main.py` accepts a video path and parse it into the functions inside `display.py` in order to process each frame, detect keypoints, match them, and then display them out.
    - Tried ORB, SIFT, and cv2.goodFeaturesToTrack for feature extraction.
    - Frames were downscaled to reduce computational load.

- **Experiments and Evaluations**:
    - **ORB (Oriented FAST and Rotated BRIEF)**:
        ![ORB Feature Detection](images/ORB.png)
        - **Observation**: The detected keypoints are spread out but seem to focus on strong edges, particularly around cars and lane markings.
        - **Strengths**:
            - Fast and efficient for real-time application.
            - Detects rotation-invariant keypoints.
            - Works well on high-contrast regions.
        - **Weaknesses**:
            - Not the best at detecting finer details.
            - Not scale-invariant (keypoints might change at different distances)

    - **SIFT (Scale-Invariant Feature Transform)**
        ![SIFT Feature Detection](images/SIFT.png)
        - **observation**: Keypoints appear to be **more distributed across the image**, especially in textured areas like trees and road signs.
        - **Strengths**:
            - Excellent at detecting **fine-grained details**.
            - Scale and rotation-invariant.
        
        - Best for **matching features across multiple frames**.
        - **Weaknesses**:
            - Computationally expensive as the observed output video is so slow. (not ideal for real-time application).

    - **goodFeaturesToTrack**:
        ![Good Features to Track](images/goodFeaturesToTrack.png)
        - **Observation**: The detected corners are **concentrated in high-contrast areas**, but a lot of weak or unnecessary points are detected when too many are allowed.
        - **Strengths**:
            - Fast and efficient for motion tracking.
            - Works well with optical flow algorithms. 
            - Tunable parameters for reducing false positives.
        - **Weaknesses**:
            - Not scale or rotation-invariant.
            - Can sometimes miss corners if parameters are not well-tuned.

    - **AKAZE (Accelerated-KAZE)**:
        ![AKAZE](images/AKAZE.png)
        - **Observation**: The detected keypoints are somewhat similar SUFT but with **faster computation time**.
        - **Strengths**:
            - Better keypoints detection.
            - Scale and rotation-invariant.
        - **Weaknesses**:
            - Slower than ORB.
            - Not as precise as SIFT.

---

### 2. Feature Detection and matching

This step extracts keypoints and descriptors from video frames and matches them across consecutive frames to track movement, a critical foundation for estimating camera motion in the SLAM pipeline.

**Detection**:

- **Implementation Methods**: Several feature detection algorithms are tested in `extractor/extractor.py`:
    - **AKAZE**: Chosen for its balance of speed and robustness. Uses `cv2.AKAZE_create()` to detect keypoints and compute binary descriptors.
    - **ORB**: Fast and lightweight, using `cv2.ORB_create()`—suitable for real-time but less robust to scale/rotation changes.
    - **SIFT**: Slower but highly robust (`cv2.SIFT_create()`), excels in varied lighting and perspectives.
    - **GoodFeaturesToTrack**: Detects strong corners (`cv2.goodFeaturesToTrack`)—simple but the output are not so ‘high-quality’, also lacks descriptors.
    
- **Current Choice:** **AKAZE is selected** for its efficiency and binary descriptors, aligning with real-time SLAM needs while handling dashcam noise (e.g., blur, lighting shifts).

- **AKAZE Details**:
    - Generates **binary descriptors** — strings of bits (0s and 1s).
    - Example: A descriptor might look like `101010...` (256 or 486 bits, depending on settings).
    - **Why binary?** Fast to compare (using Hamming distance) and memory-efficient.

- **Problem 1**: AKAZE failed to detect features in less textured regions like roads, paint strips, etc,. Making it hard to deliver high quality characteristics and uniform quantity and distribution over frames.

<figure>
    <img src="images/low quality.png" alt="Low Quality Features">
    <figcaption>AKAZE fails to capture the features in low-texture regions, also the number of detected features is limited in higher region.</figcaption>
</figure>

<p>

- **Solution**: Combine AKAZE with another detector such as ORB that also produces binary descriptors to supplement.
    - **AKAZE**: For high-quality features in textured regions.
    - **ORB**: For additional features in less textured areas.

- **Problem 2**: ORB and AKAZE combination successfully to detect much more keypoints, but still fails to capture keypoints in regions like road, lane-markings, curbs, etc,.

- **Solution**: 
    - Enhance Road Features with Preprocessing (e.g., `cv2.Canny` for edge detection) to highlight road lines and boundaries.
    - Supplement with Road-specific detectors (e.g., `cv2.goodFeaturesToTrack`) to capture corners and edges in low-texture areas.
-> By combining these detectors, the function aims to improve the number and quality of matches between consecutive frames in a dashcam video.

**Matching:**

- **Goal**: Match keypoints between frames to find correspondences (e.g., “This corner in frame 1 is here in frame 2”), enabling motion estimation.
- **Approach1**: Using OpenCV’s **Brute-Force Matcher** (`cv2.BFMatcher`) with `NORM_HAMMING` for AKAZE’s binary descriptors:
    - **What is `BFMatcher`?**
        - A brute-force matcher: Exhaustively checks all possible pairs of descriptors between two frames to find the most similar ones based on a distance metric.
        - **Distance Metric**: Hamming Distance, specifically designed for binary data.
    - **Hamming Distance**:
        - Measures the number of positions where two binary strings differ.
        - Example:
            - Descriptor 1: `1010`.
            - Descriptor 2: `1001`.
            - Hamming Distance = 2 (differs at positions 3 and 4).
        - Why using it? Because it’s efficient for binary descriptors (like AKAZE’s), counting bit mismatches directly.
    - **1-NN Matching (`match()`):**
        - Finds the nearest neighbor (smallest Hamming distance) for each descriptor in frame 1 from frame 2.
        - Parameter `crossCheck = True` ensures mutual best matches — e.g., point A matches B, and B matches A back — reducing false positives.
        - Simple and fast, but potentially give incorrect results due to affection of factors such as noise, light-quality, similar objects, etc,.
    - **K-NN Matching (`knnMatch` );**
        - Returns the top k nearest neighbors per descriptor.
        - Applies **Lowe’s ratio test**: `ratio = best_distance / second_best_distance < 0.75` keeps only distinctive matches (e.g., best is much closer than second-best).
        - Filters out ambiguous matches in noisy data (e.g., repetitive road lines or occluded objects).

- **Approach 2**: Since Brute-Force Matching can be slow due to exhaustive search, we can leverage methods like FLANN (Fast Library for Approximate Nearest Neighbors) to approximate nearest neighbor.
    - **FLANN**: A library for fast nearest neighbor search in high-dimensional spaces.
    - **Mechanism**: It builds an index of descriptors (e.g., using LSH for binary descriptors) and searches for matches efficiently. For binary descriptors, it uses Hamming distance and parameters like `table_number=6` and `key_size=12` to optimize speed and accuracy.
    - **Implementation**: 
        - `cv2.FlannBasedMatcher` is used with parameters for AKAZE descriptors.
        - Matches are filtered using Lowe’s ratio test.

- **Reducing Outliers in Matches**:
    - **RANSAC (Random Sample Consensus)**: A robust method to estimate parameters of a mathematical model from a set of observed data containing outliers.
        - **Mechanism**: It iteratively samples a minimal subset of matches (e.g., 8 matches to compute the fundamental matrix), builds a model, and counts how many other matches (inliers) fit this model within a threshold (e.g., reprojection error < 3.0 pixels). The model with the most inliers is selected, and outliers are discarded. 

    - **Lowe's ratio test**: 
        - **Purpose**: Descriptor matching can produce ambiguous matches, especially in repetitive regions like trees or sky. Lowe’s ratio test filters out unreliable matches by ensuring the best match is significantly better than the second-best.
        - **Mechanism**: For each keypoint, the matcher finds the two nearest descriptors in the other frame (using Hamming distance for binary descriptors like AKAZE/ORB). A match is kept if the distance to the closest descriptor is less than 0.65 times the distance to the second closest, indicating high confidence.
        - **Example**: A keypoint’s closest match has a Hamming distance of 30, and the second closest is 50. Since 30 < 0.65 * 50 (32.5), the match is kept; otherwise, it’s discarded as ambiguous.

<figure>
    <img src="images/Final_Matching_result.png" alt="Final Matching Result">
    <figcaption>The final matching result after combining AKAZE, ORB, and goodFeaturesToTrack. The quality and quantity of features and matches are way better than previously.</figcaption>
</figure>

<p>

**Pipeline Overview**:
The `extract_akaze_orb_features` function in `extractor.py` follows these steps:

1. **Preprocessing**: Enhance road features in both frames using `preprocess_image` (CLAHE → Gaussian blur → Canny edges → blend with original image) to make lane markings and road textures more prominent.

2. **Feature Extraction**:
- Detect AKAZE keypoints and descriptors with a threshold of 0.0001 to capture detailed features.
- Detect ORB keypoints and descriptors, limiting to 1500 features for efficiency.
- Use `goodFeaturesToTrack` to detect road-specific corners (e.g., lane markings, curbs), computing ORB descriptors for these keypoints.

3. `Keypoint Combination`: Concatenate AKAZE, ORB, and road keypoints into a single list per frame for unified processing.

4. **Feature Matching**:
- Match AKAZE, ORB, and road descriptors separately using FLANN (k=2 nearest neighbors) for speed.
- Apply Lowe’s ratio test (threshold 0.65) to filter unreliable matches.
- Adjust match indices to account for the combined keypoint lists.

5. **Outlier Filtering**: Use RANSAC to compute the fundamental matrix and filter outlier matches, ensuring geometric consistency.

6. **Return**: Output keypoints and good matches for further SLAM processing, such as motion estimation.
