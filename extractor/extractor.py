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

# Extract feature using AKAZE (Faster than SIFT but slower than ORB)
def extract_akaze_features(img):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(img, None)
    return cv2.drawKeypoints(img, keypoints, None, (0, 255, 0))

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



