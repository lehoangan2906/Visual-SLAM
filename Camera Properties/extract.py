#!/usr/bin/python3

import cv2
import random
import numpy as np


def line_intersection(line1, line2):
    """
    Function to find the intersection point of two lines.
    Used to find the vanishing points.
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    
    # Calculate the intersection point using the determinant method
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if the lines are parallel
    if abs(denom) < 1e-6:
        return None

    # Calculate the intersection point
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return (px, py)



def get_slope(line):
    """
    Function to calculate the slope of a line.
    """
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:
        return float('inf')

    return (y2 - y1) / (x2 - x1)


def find_vanishing_points(lines_group, width, height):
    """
    Function to find the vanishing points from a list of lines.

    Args:
        lines_group (list): List of lines (each line is represented as a tuple of coordinates).

    Returns:
        np.ndarray: The vanishing point (x, y) if found, otherwise None.
    """
    intersections = []

    # Find the intersection points of all pairs of lines in the group (near horizontal or near vertical)
    for i in range(len(lines_group)):
        for j in range(i + 1, len(lines_group)):
            intersection = line_intersection(lines_group[i], lines_group[j])

            # Check if the intersection point is valid (i.e., within the frame)
            if intersection and 0 <= intersection[0] < width and 0 <= intersection[1] < height:
                intersections.append(intersection)

    if not intersections:
        return None

    # Return the final point which is the median of all the intersection points in the group
    return np.mean(intersections, axis = 0)


def estimate_intrinsic_matrix(video_path, num_frames=10):
    """
    Estimate the camera's intrinsic matrixs (K) using the vanishing point method.

    Args:
        video_path (str): Path to the source video file.
        num_frames (int): Number of frames to sample for averaging.

    Returns;
        np.ndarrays: 3x3 intrinsic matrix K
    """


    # =============== Read the video file and extract frames ===============
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")   


    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        raise ValueError(f"Video has fewer than {num_frames} frames")

    
    # Sample random indices from the video
    frame_indices = random.sample(range(total_frames), num_frames)
    frames = []
    height, width = None, None

    
    # Sample random frames from the video
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Downscale frame to match display.py (1/4 of the original size)
        frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
        frames.append(frame)    # Append the frame to the list

        # Get the height and width of the first frame
        if height is None:
            height, width = frame.shape[:2]

    cap.release()    # Release the video capture object
    if not frames:
        raise ValueError("No frames were read from the video.")


    # =============== Detect the vanishing points ===============
    vanishing_points_pairs = [] # Contains pairs of vanishing points in all sampled frames

    for frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50,maxLineGap=10)        # Numpy array containing multiple lines detected in the frame

        # Skip this frame if less than 2 lines are detected
        if lines is None or len(lines) < 2:
            continue

        # =============== Cluster lines into two groups based on slope (e.g., road direction and perpendicular)
        slopes = [get_slope(line) for line in lines]
        slopes = np.array(slopes)                   # Convert into numpy array
        finite_slopes = slopes[np.isfinite(slopes)]  # Filter out infinite slopes

        # Skip this frame if less than 2 finite slopes are detected
        if len(finite_slopes) < 2:
            continue


        # Cluster into two groups (e.g., road direction and perpendicular)
        # Use a simple threshold to separate near-vertical and near-horizontal lines
        slope_threshold = np.tan(np.pi/4)   # 45 degrees
        group1 = [lines[i] for i in range(len(lines)) if abs(slopes[i]) > slope_threshold]  # Near-vertical
        group2 = [lines[i] for i in range(len(lines)) if abs(slopes[i]) <= slope_threshold] # Near-horizontal



        # Check to ensure that we can compute at least 2 vanishing points for each frame
        if len(group1) < 2 or len(group2) < 2:
            continue


        # =============== Find the intersection points (vanishing points) from lines ===============

        vp1 = find_vanishing_points(group1, width, height)
        vp2 = find_vanishing_points(group2, width, height)


        # Skip if no vanishing points are found
        if vp1 is None or vp2 is None:
            continue


        vanishing_points_pairs.append((vp1, vp2))

    if not vanishing_points_pairs:
        raise ValueError("Could not detect two vanishing points in any frame.")


    # ============== Estimate the intrinsic matrix ==============

    # Average the vanishing points across frames
    vp1s, vp2s = zip(*vanishing_points_pairs)   
    vp1 = np.mean(vp1s, axis=0)
    vp2 = np.mean(vp2s, axis=0)


    # Assume that the principal point is at the center of the image
    cx = width / 2
    cy = height / 2

    # Calculate the focal length of the camera
    f_squared = -((vp1[0] - cx) * (vp2[0] - cx) + (vp1[1] - cy) * (vp2[1] - cy))

    if f_squared <= 0:
        raise ValueError("Invalid focal length calculation.")

    f = np.sqrt(f_squared)

    fx = f
    fy = f

    # Construct the intrinsic matrix K
    K = np.array([[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]], dtype=np.float32)

    return K

if __name__ == "__main__":
    video_path = "../videos/test2.mp4"
    K = estimate_intrinsic_matrix(video_path)

    print(f"Estimated Intrinsic Matrix (K): {K}")
