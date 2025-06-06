#!/usr/bin/python3

import cv2
import numpy as np

# Function to preprocess frames to enhance Road features (lane markings, curbs, etc,.)
def preprocess_image(img):
   # convert the image to grayscale as the features are more prominent in grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement in low-texture road areas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)    # Apply CLAHE to the grayscale image

    # Apply Gaussian blur to reduce noise and smoothen the image
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Apply Canny edge detection to highlight lane markings and curbs
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Convert edges back to 3-channel for compatibility with AKAZE/ORB
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend the original image with the edge image to retain some texture
    blended = cv2.addWeighted(img, 0.5, edges_3ch, 0.5, 0.0)

    return blended, enhanced    # Return both blended (for AKAZE/ORB) and enhanced (for goodFeaturesToTrack)


# Detect road-specific keypoints using GoodFeaturesToTrack
def detect_road_corners(img_enhanced, orb=None, max_corners=500, quality_level=0.01, min_distance=10, orb_nfeatures=None):
    """
        Detect road-specific corners using goodFeaturesToTrack on the enhanced grayscale image, then compute ORb descriptors for them.
        This function is used to detect road-specific features like lane markings and curbs.

    Args:
        img_enhanced (np.ndarray): Enhanced grayscale image for corner detection.
        orb (cv2.ORB, optional): Pre-initialized ORB detector. If None, one will be created.
        max_corners (int): Maximum number of corners to detect.
        quality_level (float): Quality level for GFTT.
        min_distance (int): Minimum distance between detected corners.
        orb_nfeatures (int): Number of features to detect with ORB.

    Returns:
        keypoints (list of cv2.KeyPoint): Detected keypoints.
        descriptors (np.ndarray): ORB descriptors for those keypoints.
    """
    if orb is None:
        orb = cv2.ORB_create(nfeatures=orb_nfeatures)

    corners = cv2.goodFeaturesToTrack(
            img_enhanced, maxCorners=max_corners, 
            qualityLevel=quality_level, minDistance=min_distance
    )

    if corners is None:
        return [], np.array([], dtype=np.uint8).reshape(0, 32)

    # Convert corners to keypoints
    keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=10) for c in corners]

    # Compute ORB descriptors for these keypoints because goodFeaturesToTrack does not compute descriptors
    _, descriptors = orb.compute(img_enhanced if keypoints else None, keypoints)

    return keypoints, descriptors


# Extract feature similar to the extract_akaze_orb_features function, but without the matching part.
def extract_features_single(img, akaze_thres, orb_nfeatures):
    """
        Extract features from a single image using a combination of AKAZE, ORB, and goodfeaturestotrack.
        This function is used for the first frame only.

        Args:
            img (numpy.ndarray): The input image from which to extract features.
            akaze_thres (float): The threshold for AKAZE feature detection.
            orb_nfeatures (int): The number of features to detect with ORB.

        Returns:
            A dictionary with keypoints and descriptors per detector:
            {
                "akaze": (keypoints, descriptors),
                "orb": (keypoints, descriptors),
                "road": (keypoints, descriptors)
            }

            We need to separate them per detector to avoid confusion in the matching step.
            When combining keypoints from multiple detectors (AKAZE + ORB + road), their descriptors come from different spaces:
            - AKAZE descriptors are 61-dimensional.
            - ORB descriptors are 32-dimensional.
            - Road-specific descriptors are also 32-dimensional.
            Matching must be done per-detector to ensure correct descriptor dimensions.
    """

    # preprocess the image
    img_preprocessed, img_enhanced = preprocess_image(img)

    # Initialize AKAZE and ORB detectors
    akaze = cv2.AKAZE_create(threshold=akaze_thres, diffusivity=cv2.KAZE_DIFF_PM_G2)
    orb = cv2.ORB_create(nfeatures=orb_nfeatures)

    # ==================== Masking ====================
    mask = None     # No mask is applied in this case, but can be added if needed

    # ==================== Feature Detection ====================
    kp_akaze, des_akaze = akaze.detectAndCompute(img_preprocessed, mask)
    kp_orb, des_orb = orb.detectAndCompute(img_preprocessed, mask)
    kp_road, des_road = detect_road_corners(img_enhanced, orb, orb_nfeatures=orb_nfeatures)

    # Handle empty descriptors
    if des_akaze is None:
        des_akaze = np.array([], dtype=np.uint8).reshape(0, 61)
    if des_orb is None:
        des_orb = np.array([], dtype=np.uint8).reshape(0, 32)
    if des_road is None:
        des_road = np.array([], dtype=np.uint8).reshape(0, 32)

    return {
            "akaze": (kp_akaze or [], des_akaze),
            "orb": (kp_orb or [], des_orb),
            "road": (kp_road or [], des_road)
            }

# Extract feature using a combination of AKAZE and ORB
def extract_akaze_orb_features(img1, img2, akaze_thres, orb_nfeatures):
    # Preprocess the frames to enhance road features
    img1_preprocessed, img1_enhanced = preprocess_image(img1)
    img2_preprocessed, img2_enhanced = preprocess_image(img2)
    

    # Initialize AKAZE and ORB detectors instances
    akaze = cv2.AKAZE_create(threshold=akaze_thres, diffusivity=cv2.KAZE_DIFF_PM_G2)
    orb = cv2.ORB_create(nfeatures=orb_nfeatures)


    # ==================== Masking ====================

    # Create mask to mask off some repetitive regions (sky, grass, etc.) in some scenarios (if needed)
    # height, width = img1.shape[:2]    # Get the image's dimentions
    # mask = np.ones((height, width), dtype=np.uint8) * 255
    # sky_height = int(height * 0.35)   # Declare the masked region as the top 35% of the image
    # mask[:sky_height, :]  = 0         # Mask off the top 35% of the image
    # mask[sky_height:, :]  = 0         # Mask off the bottom 65% of the image


    # Set the mask value
    mask = None

    
    # ==================== Feature Extraction ====================

    # Separately detect keypoints and descriptors with AKAZE, ORB, and goodFeaturesToTrack
    kp1_akaze, des1_akaze = akaze.detectAndCompute(img1_preprocessed, mask)
    kp2_akaze, des2_akaze = akaze.detectAndCompute(img2_preprocessed, mask)

    kp1_orb, des1_orb = orb.detectAndCompute(img1_preprocessed, mask)
    kp2_orb, des2_orb = orb.detectAndCompute(img2_preprocessed, mask)
    
    kp1_road, des1_road = detect_road_corners(img1_enhanced, orb, orb_nfeatures)
    kp2_road, des2_road = detect_road_corners(img2_enhanced, orb, orb_nfeatures)

    # Convert all keypoints to lists for consistent concatenation
    kp1_akaze = list(kp1_akaze) if kp1_akaze is not None else []
    kp2_akaze = list(kp2_akaze) if kp2_akaze is not None else []
    kp1_orb = list(kp1_orb) if kp1_orb is not None else []
    kp2_orb = list(kp2_orb) if kp2_orb is not None else []

    # Combine the keypoints, the descriptors need to be matched separately first
    kp1 = kp1_akaze + kp1_orb + kp1_road
    kp2 = kp2_akaze + kp2_orb + kp2_road

    # ==================== Corner cases handling ====================
    
    # Handle the empty descriptors case
    if des1_akaze is None:
        des1_akaze = np.array([], dtype=np.uint8).reshape(0, 61)    # if no keypoints detected, create an empty array
    if des2_akaze is None:
        des2_akaze = np.array([], dtype=np.uint8).reshape(0, 61)
    if des1_orb is None:
        des1_orb = np.array([], dtype=np.uint8).reshape(0, 32)
    if des2_orb is None:
        des2_orb = np.array([], dtype=np.uint8).reshape(0, 32)
    if des1_road is None:
        des1_road = np.array([], dtype=np.uint8).reshape(0, 32)
    if des2_road is None:
        des2_road = np.array([], dtype=np.uint8).reshape(0, 32)
   
    # Handle the case where there are no descriptors detected
    if (des1_akaze.size == 0 and des1_orb.size == 0 and des1_road.size == 0) or (des2_akaze.size == 0 and des2_orb.size == 0 and des2_road.size == 0):
        return kp1, kp2, []

    # ===================== Feature Matching ====================
    """
    # Create a FLANN matcher for binary descriptors
    FLANN_INDEX_LSH = 6 
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level = 1)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    """
    

    # Create a brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Cross-check only enabled when using 1-NN matcher, using Hamming distance as the distance measurement.

    # Set the Lowe's ratio test threshold
    lowe_thres = 0.65

    # Compute the matches between the two frames' descriptors
    """
    # 2-NN matcher
    # Match AKAZE descriptors separately
    good_matches_akaze = []     # Placeholder for good matches
    if des1_akaze.shape[0] >= 2 and des2_akaze.shape[0] >= 2:
        matches_akaze = matcher.knnMatch(des1_akaze, des2_akaze, k=2)   # Use 2-NN matcher
        good_matches_akaze = [m[0] for m in matches_akaze if len(m) == 2 and m[0].distance < lowe_thres * m[1].distance]


    # Match ORB descriptors separately
    good_matches_orb = []     # Placeholder for good matches
    if des1_orb.shape[0] >= 2 and des2_orb.shape[0] >= 2:
        matches_orb = matcher.knnMatch(des1_orb, des2_orb, k=2)   # Use 2-NN matcher
        good_matches_orb = [m[0] for m in matches_orb if len(m) == 2 and m[0].distance < lowe_thres * m[1].distance]


    # Match road-specific descriptors separately
    good_matches_road = []
    if des1_road.shape[0] >= 2 and des2_road.shape[0] >= 2:
        matches_road = matcher.knnMatch(des1_road, des2_road, k=2)
        good_matches_road = [m[0] for m in matches_road if len(m) == 2 and m[0].distance < lowe_thres * m[1].distance]
    """

    # 1-NN matcher
    good_matches_akaze = []
    if des1_akaze.shape[0] >= 2 and des2_akaze.shape[0] >= 2:
        matches_akaze = matcher.match(des1_akaze, des2_akaze)
        matches_akaze = sorted(matches_akaze, key=lambda x: x.distance)
        good_matches_akaze = matches_akaze 

    good_matches_orb = []
    if des1_orb.shape[0] >= 2 and des2_orb.shape[0] >= 2:
        matches_orb = matcher.match(des1_orb, des2_orb)
        matches_orb = sorted(matches_orb, key=lambda x: x.distance)
        good_matches_orb = matches_orb

    good_matches_road = []
    if des1_road.shape[0] >= 2 and des2_road.shape[0] >= 2:
        matches_road = matcher.match(des1_road, des2_road)
        matches_road = sorted(matches_road, key=lambda x: x.distance)
        good_matches_road = matches_road

    # Adjust ORB match indices to account for combined keypoint list
    num_akaze_kp1 = len(kp1_akaze)
    num_akaze_kp2 = len(kp2_akaze)
    num_orb_kp1 = len(kp1_orb)
    num_orb_kp2 = len(kp2_orb)

    for match in good_matches_orb:
        match.queryIdx += num_akaze_kp1     # Offset by number of AKAZE keypoints in img1
        match.trainIdx += num_akaze_kp2     # Offset by number of AKAZE keypoints in img2
    for match in good_matches_road:
        match.queryIdx += (num_akaze_kp1 + num_orb_kp1)
        match.trainIdx += (num_akaze_kp2 + num_orb_kp2)


    # Combine the good matches from both AKAZE and ORB
    good_matches = good_matches_akaze + good_matches_orb + good_matches_road
    #print(f"Total good matches before RANSAC: {len(good_matches)}")


    # ==================== RANSAC to filter outliers ====================
    if len(good_matches) > 10:
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

        if pts1.shape[0] > 0 and pts2.shape[0] > 0:
            _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.95)
            if mask is not None:
                good_matches = [m for i, m in enumerate(good_matches) if mask[i]]
            #print(f"Good matches after RANSAC: {len(good_matches)}")


    # ==================== Return the keypoints and good matches ====================
    #print(f"Keypoints in prev_img (kp1): {len(kp1)}")
    #print(f"Keypoints in img (kp2): {len(kp2)}")
    #print(f"Good matches: {len(good_matches)}")


    return kp1, kp2, good_matches
