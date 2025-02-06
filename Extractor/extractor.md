# Explanation of the Extractor Class

This class is designed to **track feature points** across consecutive video frames, It does this in **three major steps**:

1. **Feature Detection** -> Find keypoints that are stable and trackable.
2. **Feature Description** -> Encode keypoints in a way that allows them to be recognized in the next frame.
3. **Feature Matching** -> Identify which keypoints from the previous frame correspoind to keypoints in the current frame.


