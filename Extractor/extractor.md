# 🔍 The role of This Class in Feature Tracking
The `Extractor` class is designed to **track feature points** across consecutive video frames. It does this in **three major steps**:
1. **Feature Detection** -> Find keypoints that are stable and trackable.
2. **Feature Description** -> Encode keypoints in a way that allows them to be recognized in the next frame.
3. **Feature Matching** -> Identify which keypoints from the previous frame correspond to keypoints in the current frame.

Each of these steps contributes **to maintaining visual consistency between frames**, which is crucial for **motion estimation**, and **SLAM**.

---

## 1️⃣  Feature Detection (Finding Trackable Points)

```python
feats = cv2.goodFeaturesToTrack(
    np.mean(img, axis=2).astype(np.uint8), 
    maxCorners=3000,                       
    qualityLevel=0.01,                     
    minDistance=3
)

**What this does**: 
- Detect **high-contrast**, **corner-like structures** in the image.
- Ensures that the detected features are **well-distributed** and **trackable** (i.e., they have strong edges and textures that remain consistent across frames).

**Why is this important for tracking**?
- If we only used ORB's built-in keypoint detection, the features might be **too clustered** or biased toward certain areas.
- Using `goodFeaturesToTrack` **ensures that keypoints are spatially spread out**, reducing drift in tracking.

**Tracking Aspect**:
- If a feature is **reliably detected in frame** `t` **and persist in frame** `t+1`, it helps establishing a **correspondence between the two frames**, allowing us to estimate motion.

---
## 2️⃣  Feature Description (Encoding Keypoints for Recognition)

```python
kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
kps, des = self.orb.compute(img, kps)
```

**What this does**:
- Converts the detected corners into **OpenCV** `KeyPoint` **objects**, which store additional metadata (like orientation and scale).
- Uses **ORB (Oriented FAST and Rotated BRIEF)** to generate a **binary descriptor** for each keypoint.

**Why is this important for tracking**?
- Descriptors allow us to **uniquely identify** each keypoint.
- Even if a keypoint's **position shifts slightly**, the descriptor should remain similar, enabling **robust matching across frames**.

**Tracking Aspect**:
- Instead of just **matching raw coordinates** (which can be unreliable due to camera movement, occlusion, etc,.), we **match descriptors**, ensuring a feature point in frame `t` is correctly linked to the same physical feature in frame `t+1`.

---

## 3️⃣  Feature Matching (Tracking Keypoints Across Frames)

```python
if self.last is not None and self.last["des"] is not None:
    matches = self.bf.knnMatch(des, self.last["des"], k=2)  # Match descriptors of the current frame with the previous frame
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            ret.append((kps[m.queryIdx], self.last["kps"][m.trainIdx]))
```

**What this does**:
- Uses **KNN Matching** (k=2) to find the **two closest** descriptor matches for each keypoint.
- Applies **Lowe's Ratio Test** (`m.distance < 0.75 * n.distance`) to dilter out ambiguous matches.
- Stores **only the best matches** that confidently correspond to the same feature in both frames.

**Why is this important for tracking**?
- The **ratio test eliminates false matches**, ensuring only **reliable** correspondences are used.
- Tracking **good matches** allow us to establish **motion** between frames.

**Tracking Aspect**:
- The retained matches tell us **where the feature points moved** in the new frame, providing information on **camera motion**, **object movement**, and **scene structure**.

---
## 4️⃣  Storing Data for the Next Frame (Maintaining Continuity in Tracking)
```python
self.last = {"kps": kps, "des": des}
```

**What this does**:
- Stores the **current frame's keypoints and descriptors** so they can be used for matching in the next frame.

**Why is this important for tracking**?
- Without storing previous keypoints, **Tracking would reset at every frame**, making it impossible to track movement **over time**.

**Tracking Aspect**:
- The retained descriptors allow **frame** `t+1` **to be compared with frame** `t`, forming a continuous tracking process.

---
### 🔗 How This Enables Frame-by-Frame Tracking
The entire pipeline works together **to track keypoints from frame to frame**;
1. Frame `t` (first frame):
    - Detect features (`goodFeaturesToTrack`)
    - compute ORB descriptors.
    - No matching (since there's no previous frame).
    - Store keypoints/descriptors.
2. Frame `t+1`:
    - Detect new features.
    - Compute ORB descriptors.
    - Match descriptors to frame `t`.
    - Find movement of keypoints (feature tracking).
    - Store updated keypoints/descriptors.
3. Frame `t+2` and beyond:
    - Detect new features.
    - Compute ORB descriptors.
    - Match descriptors to frame `t+1`.
    - Find movement of keyponts.
    - Store updated keypoints/descriptors.
    - **Feature matches tell us how points move across frames**, revealing motion patterns.

---
### 🔍 Summary (What this code actually does for Tracking)
1. Find feature points in each frame that are likely to remain stable.
2. Encodes feature points using ORB descriptors to ensure they can be recognized later.
3. Matches feature points between frames to track where they move.
4. Filter out bad matches using KNN and Lowe's ratio test to improve tracking accuracy.
5. Links keypoints across frames, providing the basis for estimating motion.
