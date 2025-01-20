purpose: build something that will recover a map from a video recording a car moving (on dash cam)

Video has many trees → good features to track frame-by-frame

use OpenCV cv2 for displaying the video

this is a feature-based slam → track the features from frame to frame

use OpenCV orb features to identify key-points and compute their descriptions in each frame

the default setting of ORB output key-points kind of crappy because they are not well distributed across the frame. → Need to write custom orb extractor. → by breaking the frame into a grid and then detect and compute the key-points in each grid, then shift their coordinates to the image global coordinate system. → The newly detected key-points are still clustered → need to choose good features to track using OpenCV goodFeaturesToTrack. → goodFeatureToTrack only returns corners’ locations, whereas SLAM algorithms require **descriptors** (such as SIFT, ORB, or SURF) to match features between frames robustly.
