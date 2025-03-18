#!/usr/bin/python3

import cv2

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
    akaze = cv2.AKAZE_create()  # Create an AKAZE instance
    
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
