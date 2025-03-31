### Updated Summary of Progress and Issues Encountered Today

#### Problems Encountered
1. **Static Frame Issue and Matching Noise**: The previous frame (`prev_img`) wasn’t updating, causing matches to compare against the first frame, leading to a drop in matches (from 2008 to ~200) and high matching noise (e.g., incorrect horizontal lines across the road).
2. **Keypoint Clustering in Sky**: In highway footage, keypoints and matches clustered in the sky region, which lacks distinctive features for SLAM.
3. **Low Keypoints on Road**: Feature detectors (AKAZE and ORB) consistently missed features on the road, road markings, and curbs, focusing instead on cars, objects, or buildings due to the road’s low texture.
4. **Match Count Sensitivity to Parameters**:
   - Lowering AKAZE threshold (e.g., to 0.0001) reduced matches significantly due to poor descriptor quality.
   - Increasing ORB `nfeatures` to 2000 didn’t improve matches and sometimes reduced them.
   - In urban footage, AKAZE threshold above 0.01 resulted in no matches, while below 0.0005 reduced matches; ORB `nfeatures` at 2000 also reduced matches.

#### Solutions Tried
1. **Fixed Static Frame Issue and Reduced Noise**: Updated `process_frame` in `display.py` to ensure `prev_img` updates after every frame, stabilizing match counts (now ~635–809 in urban footage) and reducing matching noise to nearly 0%.
2. **Sky Masking**: Added a mask to exclude the top 30–45% of the frame in highway footage, shifting keypoints to lower regions (cars/objects), but later removed it for urban footage due to less repetitive sky patterns.
3. **Lower Region Masking**: Explored masking the bottom 20% (car hood) to focus on the road, but this wasn’t implemented in the final code.
4. **Parameter Tuning**:
   - AKAZE threshold: Tested values from 0.0001 to 0.01; settled on 0.008 for stable matches (~635–809).
   - ORB `nfeatures`: Tested up to 2000, but settled on 1500 as higher values reduced matches.
5. **RANSAC and Matching**: Kept RANSAC with a reprojection threshold of 3.0 and Lowe’s ratio test at 0.75, ensuring geometric consistency.

#### Current State
- The pipeline uses AKAZE (threshold 0.008) and ORB (`nfeatures` 1500) without masking, achieving stable matches (635–809) in urban footage with near-zero noise.
- Matches are distributed across buildings, trees, and some road features, but feature detectors still miss road markings and curbs, indicating a need for road-specific detection improvements.
- The setup is ready for the next SLAM steps (e.g., motion estimation), though road feature detection needs further attention.
