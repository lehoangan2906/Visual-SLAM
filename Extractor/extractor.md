# Explanation of the Extractor Class

## The Role of this class in Feature Tracking:

This class is designed to **track feature points** across consecutive video frames, It does this in **three major steps**:

1. **Feature Detection** -> Find keypoints that are stable and trackable.
2. **Feature Description** -> Encode keypoints in a way that allows them to be recognized in the next frame.
3. **Feature Matching** -> Identify which keypoints from the previous frame correspoind to keypoints in the current frame.

Each of these steps contributes **to maintaining visual consistency between frames**, which is cruvial for **motion estimation**, **structure-from-motion (SfM)**, and **SLAM***.

### 1. Feature Detection (Finding Trackable Points):

```python
feats = cv2.goodFeaturesToTrack(
    np.mean(img, axis=2).astype(np.uint8), 
    maxCorners=3000,                       
    qualityLevel=0.01,                     
    minDistance=3
)
```

💡 **What this does:**
- Detects **high-contrast**, **corner-like structures** in the image.
- Ensures that the detected features are **well-distributed** and **trackable** (i.e., they have strong textures that remain consistent across frames).

📌 **Why is this important for tracking?**
- If we only used ORB's built-in keypoint detection, the features might be **too clustered** or biased toward certain areas.
- Using 'goodFeaturesToTrack' **ensures that keypoints are spatially spread out**, reducing drift in tracking.


