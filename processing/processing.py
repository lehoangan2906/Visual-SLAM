#!/usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from extractor.extractor import extract_akaze_orb_features

# Global variables to store the previous frame and its state
prev_img = None
prev_kp = None
prev_good_matches = None
camera_poses = []  # List to store camera poses (R, t) for each frame
map_points = []    # List to store 3D points and their associations


def normalize_coordinates(pts, K):
    """
    Convert pixel coordinates to normalized camera coordinates using K^-1.

    Args:
        pts: Array of shape (N, 2) containing pixel coordinates [x, y].
        K: 3x3 intrinsic matrix.

    Returns:
        Array of shape (N, 2) containing normalized coordinates [u, v].
    """
    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Convert to homogeneous coordinates
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Shape: (N, 3)

    # Compute K^-1
    K_inv = np.array([[1/fx, 0, -cx/fx],
                      [0, 1/fy, -cy/fy],
                      [0, 0, 1]])

    # Normalize: x̂ = K^-1 * x
    pts_norm_h = (K_inv @ pts_h.T).T  # Shape: (N, 3)
    pts_norm = pts_norm_h[:, :2] / pts_norm_h[:, 2:3]  # Divide by last coordinate, shape: (N, 2)

    return pts_norm


def compute_essential_matrix(pts1, pts2):
    """
    Compute the essential matrix using the 8-point algorithm.

    Args:
        pts1, pts2: Arrays of shape (N, 2) containing normalized coordinates [u, v].

    Returns:
        E: 3x3 essential matrix, or None if computation fails.
    """
    if len(pts1) < 8:
        print("Not enough matches (< 8) to compute essential matrix.")
        return None

    # Construct the matrix A for the epipolar constraint: x̂_2^T E x̂_1 = 0
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]

    # Solve A e = 0 using SVD
    _, _, V = np.linalg.svd(A)
    e = V[-1]  # Last column of V (smallest singular value)
    E = e.reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, V = np.linalg.svd(E)
    S = np.array([S[0], S[1], 0])  # Set smallest singular value to 0
    E = U @ np.diag(S) @ V

    # Normalize E to have unit norm (optional, for numerical stability)
    E /= np.linalg.norm(E)

    return E


def decompose_essential_matrix(E, pts1, pts2, K):
    """
    Decompose E into R and t, selecting the solution with positive depth.

    Args:
        E: 3x3 essential matrix.
        pts1, pts2: Arrays of shape (N, 2) containing pixel coordinates [x, y].
        K: 3x3 intrinsic matrix.

    Returns:
        R: 3x3 rotation matrix.
        t: 3x1 translation vector.
        Or (None, None) if decomposition fails.
    """
    if E is None:
        return None, None

    # Perform SVD
    U, _, V = np.linalg.svd(E)

    # Ensure proper orientation
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(V) < 0:
        V *= -1

    # Define W matrix for rotation
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # Four possible solutions
    R1 = U @ W @ V.T
    R2 = U @ W.T @ V.T
    t1 = U[:, 2].reshape(3, 1)  # Third column of U
    t2 = -t1

    # Convert a few points to normalized coordinates for triangulation
    pts1_norm = normalize_coordinates(pts1[:5], K)
    pts2_norm = normalize_coordinates(pts2[:5], K)

    # Test each solution by triangulating a few points
    def check_positive_depth(R, t, pts1_n, pts2_n):
        # Camera matrices
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # First camera: [I | 0]
        P2 = np.hstack([R, t])  # Second camera: [R | t]

        points_3d = []
        for p1, p2 in zip(pts1_n, pts2_n):
            p1_h = np.array([p1[0], p1[1], 1])
            p2_h = np.array([p2[0], p2[1], 1])

            # Construct system A * X = 0
            A = np.zeros((4, 4))
            # For first camera: u1 * (P1_3 * X) - (P1_1 * X) = 0, v1 * (P1_3 * X) - (P1_2 * X) = 0
            A[0] = p1_h[1] * P1[2] - P1[1]  # v1 * P1_3 - P1_2
            A[1] = P1[0] - p1_h[0] * P1[2]  # P1_1 - u1 * P1_3
            # For second camera: u2 * (P2_3 * X) - (P2_1 * X) = 0, v2 * (P2_3 * X) - (P2_2 * X) = 0
            A[2] = p2_h[1] * P2[2] - P2[1]  # v2 * P2_3 - P2_2
            A[3] = P2[0] - p2_h[0] * P2[2]  # P2_1 - u2 * P2_3

            # Solve using SVD
            _, _, V = np.linalg.svd(A)
            X = V[-1]
            X /= X[3]  # Normalize homogeneous coordinate
            points_3d.append(X[:3])

        # Check depth
        points_3d = np.array(points_3d)
        z1 = points_3d[:, 2]  # Depth in first camera
        P2_points = (P2 @ np.hstack([points_3d, np.ones((len(points_3d), 1))]).T).T
        z2 = P2_points[:, 2]  # Depth in second camera
        return np.sum((z1 > 0) & (z2 > 0))

    # Evaluate all four solutions
    solutions = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    best_score = -1
    best_R, best_t = None, None
    for R, t in solutions:
        score = check_positive_depth(R, t, pts1_norm, pts2_norm)
        if score > best_score:
            best_score = score
            best_R, best_t = R, t

    return best_R, best_t


def triangulate_points(pts1_norm, pts2_norm, R, t):
    """
    Triangulate 3D points from matched points in two frames.

    Included steps:
        - Set up the camera projection matrices
        - Formulate the triangulation equation for each 3D point
        - Solve for each 3D point X^i using SVD
        - Filter out invalid 3D points

    Args: 
        - pts1_norm: normalized coordinates [u, v] for frame t.
        - pts2_norm: normalized coordinates [u, v] for frame t+1.
        - R: 3x3 rotation matrix (from frame t to frame t+1).
        - t: 3x1 translation vector (from frame t to frame t+1).

    Returns:
        - points_3d: Array of shape (N, 3) containing 3D points [X, y, Z].
        - valid_mask: Boolean array indicating valid points (positive depth).
    """

    
    points_3d = []
    valid_mask = []

    
    # Projection matrices
    P1 = np.hstack([np.eye(3), np.zeros((3,1))])    # [I|0] for the first frame
    P2 = np.hstack([R, t])                          # [R|t] for the second frame


    # Triangulate each pair of points
    for p1, p2 in zip(pts1_norm, pts2_norm):
        p1_h = np.array([p1[0], p1[1], 1])  # Homogeneous: [u1, v1, 1]
        p2_h = np.array([p2[0], p2[1], 1])  # Homogeneous: [u2, v2, 1]

        # Construct the system of equation A*X = 0
        A = np.zeros((4, 4))
        A[0] = p1_h[1] * P1[2] - P1[1]  # v1 * P1_3 - P1_2
        A[1] = P1[0] - p1_h[0] * P1[2]  # P1_1 - u1 * P1_3
        A[2] = p2_h[1] * P2[2] - P2[1]  # v2 * P2_3 - P2_2
        A[3] = P2[0] - p2_h[0] * P2[2]  # P2_1 - u2 * P2_3
    
        # Solve using SVD
        _, _, V = np.linalg.svd(A)
        X = V[-1]   # The solution X is the right singular vector corresponding to the smallest singular value (last column of V)
        X /= X[3]   # Normalize homogeneous coordinate to make the last coordinate W to be 1  (X = [X, Y, Z, W] -> X = [X, Y, Z, 1])
        X_3d = X[:3] # X = [X, Y, Z]


        # Check the depth in both views:
        z1 = X_3d[2]    # Depth in the first camera (Z coordinate value)
        X2_h = P2 @ X   # Project to the second camera
        z2 = X2_h[2]    # Depth in the second camera (Z coordinate value)


        # Return valid if depth is positive in both views
        is_valid = (z1 > 0) and (z2 > 0)
        points_3d.append(X_3d if is_valid else np.zeros(3))
        valid_mask.append(is_valid)


    points_3d = np.array(points_3d)
    valid_mask = np.array(valid_mask, dtype=bool)


    return points_3d, valid_mask


def visualize_map():
    """
    Visualize the 3D map and camera trajectory using the stored 3D points and camera poses.
    """

    # Extract 3D points from map_points
    points_3d = np.array([entry['point_3d'] for entry in map_points])


    # Compute global camera poses by accumulating relative poses
    global_poses = [(np.eye(3), np.zeros((3, 1)))]  # Start with identity for the first frame
    for R, t in camera_poses[1:]:
        prev_R, prev_t = global_poses[-1]
        new_R = prev_R @ R
        new_t = prev_t + prev_R @ t
        global_poses.append((new_R, new_t))


    # Extract camera positions
    camera_positions = np.array([pose[1].flatten() for pose in global_poses])


    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    if len(points_3d) > 0:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c = 'b', s=1, label = '3D Points')

    else:
        print("No 3D points to visualize.")

    # Plot camera trajectory
    if len(camera_positions) > 0:
        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 'r-', label='Camera_Trajectory')
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='r', s=50, label='Camera Positions')
    else:
        print("No camera poses to visualize.")

    # Set labels and legent
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('3D Map and Camera Trajectory')
    plt.show()


# Reads a video, processes each frame, and displays the results.
def play_video(video_path, K):
    """
    Play the video from the given path, processing each frame to extract and display features.

    Args:
        video_path (str): The path to the video file.
        K (np.ndarray): 3x3 camera intrinsic matrix.
    """
    global camera_poses

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Could not open the video file: ", video_path)
        return

    # Read and process frames until the video ends
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break       # End of video

        # Increment frame counter
        frame_count += 1

        # Downscale the frame to reduce computational load
        frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))

        # Ensure the frame is in color (BGR) for visualization
        if len(frame.shape) == 2:   # If grayscale, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Process the frame (extract features, estimate pose, and display)
        process_frame(frame, K)

        # Press 'q' to quit the video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


    # Visualize the 3D map and camera trajectory
    visualize_map()


# Extracts features, matches keypoints with the previous frame, estimates pose, and visualizes matches.
def process_frame(img, K):
    """
    Process a single frame to extract features, match with the previous frame, estimate camera pose,
    and display the results.

    Args:
        img (numpy.ndarray): The input frame to process.
        K (np.ndarray): 3x3 camera intrinsic matrix.
    """
    global prev_img, prev_kp, prev_good_matches, camera_poses, map_points


    # Initialize on the first frame
    if prev_img is None:
        prev_img = img   # Store the current frame for the next iteration
        
        # Initialize prev_kp without matching
        akaze = cv2.AKAZE_create(threshold=0.008, diffusivity=cv2.KAZE_DIFF_PM_G2)
        orb = cv2.ORB_create(nfeatures=1500)
        
        # Preprocess the image (same as in extract_akaze_orb_features)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        img_preprocessed = cv2.addWeighted(img, 0.5, edges_3ch, 0.5, 0.0)

        # Detect keypoints with AKAZE and ORB
        kp_akaze, _ = akaze.detectAndCompute(img_preprocessed, None)
        kp_orb, _ = orb.detectAndCompute(img_preprocessed, None)
        prev_kp = kp_akaze + kp_orb if kp_akaze and kp_orb else (kp_akaze or kp_orb) or []
        cv2.imshow("SLAM Output", prev_img)
        cv2.waitKey(1)

        # Initialize the first camera pose as identity
        R = np.eye(3, dtype=np.float32)
        t = np.zeros((3, 1), dtype=np.float32)
        camera_poses.append((R, t))
        return

    # Extract keypoints and matches between the previous and current frame
    kp1, kp2, good_matches = extract_akaze_orb_features(prev_img, img)

    # Estimate camera pose from scratch
    if len(good_matches) >= 8:
        # Extract pixel coordinates
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

        # Convert to normalized coordinates
        pts1_norm = normalize_coordinates(pts1, K)
        pts2_norm = normalize_coordinates(pts2, K)

        # Compute essential matrix
        E = compute_essential_matrix(pts1_norm, pts2_norm)

        # Decompose into R and t
        R, t = decompose_essential_matrix(E, pts1, pts2, K)

        if R is not None and t is not None:
            camera_poses.append((R, t))
            print(f"Frame {len(camera_poses)} - Rotation Matrix:\n{R}")
            print(f"Frame {len(camera_poses)} - Translation Vector:\n{t}")


            # Triangulate 3D points
            points_3d, valid_mask = triangulate_points(pts1_norm, pts2_norm, R, t)


            # Store valid 3D points with their associations
            current_frame_idx = len(camera_poses) - 1   # Frame t
            prev_frame_idx = current_frame_idx - 1      # Frame t-1

            for i, is_valid in enumerate(valid_mask):
                if is_valid:
                    point_3d = points_3d[i]

                    # Store associations: (3D point, [frame_idx1, frame_idx2], [keypoint_idx1, keypoint_idx2])
                    associations = {
                        "point_3d": point_3d,
                        "frame_indices": [prev_frame_idx, current_frame_idx],
                        "keypoint_indices": [good_matches[i].queryIdx, good_matches[i].trainIdx]
                            }

                    map_points.append(associations)

                print(f"Triangulated {np.sum(valid_mask)} valid 3D points out of {len(good_matches)} matches.")
        else:
            # Fallback: assume no motion
            R = np.eye(3, dtype=np.float32)
            t = np.zeros((3, 1), dtype=np.float32)
            camera_poses.append((R, t))
            print("Using identity pose as fallback.")
            
    else:
        # Not enough matches
        R = np.eye(3, dtype=np.float32)
        t = np.zeros((3, 1), dtype=np.float32)
        camera_poses.append((R, t))
        print("Not enough matches; using identity pose.")

    # Visualize the results
    if good_matches and len(kp1) > 0 and len(kp2) > 0:
        # Create an output image for drawing matches
        height = max(prev_img.shape[0], img.shape[0])
        width = prev_img.shape[1] + img.shape[1]
        output_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw the matches between the two frames
        img_matches = cv2.drawMatches(
            prev_img, kp1, img, kp2, good_matches, output_img,
            matchColor=(0, 255, 0),   # Green lines for matches
            singlePointColor=(0, 0, 255), # Red dots for keypoints
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow("SLAM Output", img_matches)
    else:
        # Fallback: display the current frame with keypoints
        img_with_kp = img.copy()
        if len(kp2) > 0:
            img_with_kp = cv2.drawKeypoints(
                img, kp2, img_with_kp,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        cv2.imshow("SLAM Output", img_with_kp)

    # Update the previous frame for the next iteration
    prev_img = img.copy()
    prev_kp = kp2
    prev_good_matches = good_matches


if __name__ == "__main__":
    video_path = "videos/test.mp4"
    # For testing, use a dummy K matrix
    K = np.array([[500, 0, 320],
                  [0, 500, 240],
                  [0, 0, 1]], dtype=np.float32)
    play_video(video_path, K)
