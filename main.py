#!/usr/bin/python3

import numpy as np
from processing.processing import play_video
from camera_properties.extract import estimate_intrinsic_matrix

def estimating_intrinsic_matrix(video_path):
    """
    Estimate the camera's intrinsic matrix using the vanishing point method.
    Stablilize the resulting K matrix by averaging over multiple runs to reduce outliers.
    """
    
    print("----------------------------------------------------------\n\n        Estimating the Camera's Intrinsic Matrix K\n\n                This may take a moment\n\n----------------------------------------------------------")

    # Compute the intrinsic matrix K of the camera by averaging multiple runs
    num_runs = 8 # Number of runs to average
    
    fx_list, fy_list, cx_list, cy_list = [], [], [], []     # Lists to store vanishing points and focal lengths

    for _ in range(num_runs):
        K = estimate_intrinsic_matrix(video_path, num_frames = 20, verbose = False)
        
        if K is not None:
            fx_list.append(K[0, 0])
            fy_list.append(K[1, 1])
            cx_list.append(K[0, 2])
            cy_list.append(K[1, 2])
        else:
            num_runs+=1

    # Take the median to reduce the impact of outliers
    fx = np.median(fx_list)
    fy = np.median(fy_list)
    cx = np.median(cx_list)
    cy = np.median(cy_list)

    # Reconstruct the intrinsic matrix K
    K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)

    print(f"Intrinsic matrix K:\n{K}\n")

    return K

if __name__ == "__main__":
    video_path = "videos/test2.mp4"
    
    # Estimate the camera's intrinsic matrix first
    K = estimating_intrinsic_matrix(video_path)


    # Use the play_video function from display.py 
    play_video(video_path, K)

