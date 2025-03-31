#!/usr/bin/python3

import cv2
import numpy as np


# Extract feature using a combination of AKAZE and ORB
def extract_akaze_orb_features(img1, img2):
    # Initialize AKAZE and ORB instances
    akaze = cv2.AKAZE_create(threshold=0.008, diffusivity=cv2.KAZE_DIFF_PM_G2)  
    orb = cv2.ORB_create(nfeatures=1500)

    
    # Since the keypoints detected are clustering around the sky region, we need to create a mask to exclude that region (top 30% of the image)
    #height, width = img1.shape[:2]
    #mask = np.ones((height, width), dtype=np.uint8) * 255
    #sky_height = int(height * 0.35)  # Mask the top of the image
    #mask[:sky_height, :] = 0         # Set the top region to 0 (exclude)
    # mask[sky_height:, :] = 0        # Set the bottom region to 0 (exclude)

    # Detect keypoints and their corresponding descriptors with AKAZE
    kp1_akaze, des1_akaze = akaze.detectAndCompute(img1, mask=None)
    kp2_akaze, des2_akaze = akaze.detectAndCompute(img2, mask=None)

    # Detect keypoints and descriptors with ORB
    kp1_orb, des1_orb = orb.detectAndCompute(img1, mask=None)
    kp2_orb, des2_orb = orb.detectAndCompute(img2, mask=None)


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

    # Create a brute-force matcher instance
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
    print(f"Total good matches before RANSAC: {len(good_matches)}")


    # Apply RANSAC to filter matches (if enough matches exist)
    if len(good_matches) > 10:
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

        if pts1.shape[0] > 0 and pts2.shape[0] > 0:
            _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.95)
            if mask is not None:
                good_matches = [m for i, m in enumerate(good_matches) if mask[i]]
            print(f"Good matches after RANSAC: {len(good_matches)}")


    print(f"Keypoints in prev_img (kp1): {len(kp1)}")
    print(f"Keypoints in img (kp2): {len(kp2)}")
    print(f"Good matches: {len(good_matches)}")

    return kp1, kp2, good_matches

