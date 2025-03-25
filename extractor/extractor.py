#!/usr/bin/python3

import cv2
import numpy as np

# Extract feature using ORB (fast, suitable for real-time)
def extract_orb_features(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)

    # Draw keypoints on the image
    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    return img_keypoints


# Extract feature using SIFT (slower but robust against scale, rotation, lighting changes)
def extract_sift_features(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255,0))

    return img_keypoints


# Extract feature using cv2.GoodFeaturesToTrack
# Find the strongest corners in an image
def extract_good_features(img, max_corners=500, quality_level=0.04, min_distance=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the top 100 corners
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)

    if corners is not None:
        corners = corners.astype(int)  # Convert to integer

        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)     # Draw circles on detected keypoints

    return img


# Extract feature using Accelerated KAZE (Faster than SIFT, slower than ORB but give the most stable keypoints output)
def extract_akaze_features(img1, img2):
    akaze = cv2.AKAZE_create(threshold=0.001)  # Create an AKAZE instance
    
    # Detect keypoints and their corresponding descriptors in each frame
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # Create a matcher instance 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Compute the matches between the two frames' descriptors
    # good_matches = matcher.match(des1, des2) # Brute-force matching (1 Nearest Neighbor matching)

    # Sort by distance
    # matches = sorted(matches, key=lambda x: x.distance)

    matches = matcher.knnMatch(des1, des2, k=2) # Knn match for finding the top k matches for each descriptor instead of just 1.

    good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]  # Lowe's ratio test to filter out best matches (check if the best match is significantly better than the second best match)
    
    return kp1, kp2, good_matches


# Extract feature using a combination of AKAZE and ORB
def extract_akaze_orb_features(img1, img2):
    # Initialize AKAZE and ORB instances
    akaze = cv2.AKAZE_create(threshold=0.0005, diffusivity=cv2.KAZE_DIFF_PM_G2)  
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=12)


    # Detect keypoints and their corresponding descriptors with AKAZE
    kp1_akaze, des1_akaze = akaze.detectAndCompute(img1, None)
    kp2_akaze, des2_akaze = akaze.detectAndCompute(img2, None)

    # Detect keypoints and descriptors with ORB
    kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
    kp2_orb, des2_orb = orb.detectAndCompute(img2, None)


    # Handle empty descriptors
    if des1_akaze is None:
        des1_akaze = np.array([], dtype=np.uint8).reshape(0, 61)    # if no keypoints detected, create an empty array
    if des2_akaze is None:
        des2_akaze = np.array([], dtype=np.uint8).reshape(0, 61)
    if des1_orb is None:
        des1_orb = np.array([], dtype=np.uint8).reshape(0, 32)
    if des2_orb is None:
        des2_orb = np.array([], dtype=np.uint8).reshape(0, 32)


    # After detecting keypoints and descriptors, combine them into one unified set.
    kp1 = kp1_akaze + kp1_orb
    kp2 = kp2_akaze + kp2_orb


    # Handle the case where there are no descriptor detected
    if (des1_akaze.size == 0 or des1_orb.size == 0) or (des2_akaze.size == 0 or des2_orb.size == 0):
        return kp1, kp2, [] # Return keypoints and empty matches if no descriptors detected 

    # Create a matcher instance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Crosscheck only enabled when using 1-NN matcher

    # Set the Lowe's ratio test threshold
    lowe_thres = 0.75

    # Compute the matches between the two frames' descriptors
    # Match AKAZE descriptors separately
    good_matches_akaze = []
    if des1_akaze.size > 0 and des2_akaze.size > 0:
        matches_akaze = matcher.knnMatch(des1_akaze, des2_akaze, k = 2)
        good_matches_akaze = [m[0] for m in matches_akaze if len(m) == 2 and m[0].distance < lowe_thres * m[1].distance]
        # good_matches_akaze = [m[0] for m in matches_akaze if len(m) >= 2]   # no ratio test for debugging

    # Match ORB descriptors separately
    good_matches_orb = []
    if des1_orb.size > 0 and des2_orb.size > 0:
        matches_orb = matcher.knnMatch(des1_orb, des2_orb, k = 2)
        good_matches_orb = [m[0] for m in matches_orb if len(m) == 2 and m[0].distance < lowe_thres * m[1].distance]
        #good_matches_orb = [m[0] for m in matches_orb if len(m) >= 2]   # no ratio test for debugging

    # Adjust ORB match indices to account for combined keypoint list
    num_akaze_kp1 = len(kp1_akaze)
    num_akaze_kp2 = len(kp2_akaze)
    for m in good_matches_orb:
        m.queryIdx += num_akaze_kp1  # Offset by number of AKAZE keypoints in img1
        m.trainIdx += num_akaze_kp2  # Offset by number of AKAZE keypoints in img2

    # Combine the matches
    good_matches = good_matches_akaze + good_matches_orb


    return kp1, kp2, good_matches
