#!/usr/bin/python3

import numpy as np
from processing.processing import play_video

if __name__ == "__main__":
    video_path = "videos/test3.MOV"
    
    K = np.array([[3612, 0, 1920],
                  [0, 2372, 1080],
                  [0, 0, 1]], dtype=np.float32)
    
    # Use the play_video function from display.py 
    play_video(video_path, K)

