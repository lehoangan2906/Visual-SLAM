import cv2
import numpy as np

class Extractor(object):
    GX = 8
    GY = 6

    def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img):
        # Detection
        # =========
        # Detect strong corners (features)
        feats = cv2.goodFeaturesToTrack(
            np.mean(img, axis=2).astype(np.uint8), # Convert the image to grayscale
            maxCorners=3000,                       # Maximum number of corners to return
            qualityLevel=0.01,                     # Minimum quality of corners
            minDistance=3                          # Minimum Euclidean distance between corners
        )

        # Extraction
        # ==========
        # Convert detected corners to KeyPoint objects
        kps = [cv2.KeyPoint(x=f[0][0], y = f[0][1], size=20) for f in feats]    # Extract the x and y coordinates of the corner then assign a fixed size of 20 pixels for each keypoint.

        # Compute the keypoint descriptor with ORB
        kps, des = self.orb.compute(img, kps)

        # Matching
        # ========
        ret = []
        # Match descriptors with the previous frame if available
        if self.last is not None and self.last["des"] is not None:
            matches = self.bf.knnMatch(des, self.last["des"], k = 2) # Match the descriptors of the current frame with the descriptors of the previous frame
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    ret.append((kps[m.queryIdx], self.last["kps"][m.trainIdx]))

        # Store the current frame's keypoints and descriptors for use in the next frame
        self.last = {"kps": kps, "des": des}

        return ret