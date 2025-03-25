#!usr/bin/python3

import cv2
import numpy as np 
from extractor.extractor import extract_akaze_orb_features

# Global variable to store the previous frame and its state
prev_img = None


# For displaying the video
def play_video(video_path):
    """
    Play the video from the given path, processing each frame to extract and display features.

    Args:
        video_path (str): The path to the video file.
    """

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

        # Increment frame counter (for debugging or frame skipping if needed)
        frame_count += 1

        # Downscale the frame to reduce computational load
        frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))

        # Ensure the frame is in color (BGR) for visualization
        if len(frame.shape) == 2:   # If grayscale, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Process the frame (extract features and display)
        process_frame(frame)

        # Display the frame number for debugging (if needed)
        # print("Frame: ", frame_count)

        # Press 'q' to quit the video 
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


# For processing each frames
def process_frame(img):
    """
    Process a single frame to extract features, match with the previous frame, and 
    display the results.

    Args:
        img (numpy.ndarray): The input frame to process.
    """

    global prev_img


    # Initialize on the first frame
    if prev_img is None:
        prev_img = img.copy()   # Store the current frame for the next iteration
        cv2.imshow("SLAM Output", prev_img)
        cv2.waitKey(1)
        return

    # Extract keypoints and matches between the previous and current frame
    kp1, kp2, good_matches = extract_akaze_orb_featuers(prev_img, img)


    # Debugging: Print keypoint and match counts
    print(f"Keypoints in prev_img (kp1): {len(kp1)}")
    print(f"Keypoints in img (kp2): {len(kp2)}")
    print(f"Good matches: {len(good_matches)}")


    # Visualize the results
    if good_matches and len(kp1) > 0 and len(kp2) > 0:
        # Create an output image for drawing matches
        height = max(prev_img.shape[0], img.shape[0])
        width = prev_img.shape[1] + img.shape[1]
        output_img = np.zeros((height, width, 3), dtype=np.uint8)


        # Draw the matches between the two frames on the output image
        img_matches = cv2.drawMatches(
            prev_img, kp1, img, kp2, good_matches, output_img,
            matchColor = (0, 255, 0),   # Green lines for matches
            singlePointColor = (0, 0, 255), # Red dots for keypoints
            flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

        cv2.imshow("SLAM Output", img_matches)

    else:
        # Fallback: display the current frame with keypoints if there are no matches
        img_with_kp = img.copy()
        if len(kp2) > 0:
            img_with_kp = cv2.drawKeypoints(
                    img, kp2, img_with_kp,
                    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )

            cv2.imshow("SLAM Output", img_with_kp)

        # Update the previous frame for the next iteration
        prev_img = img.copy()

if __name__ == "__main__":
    video_path = "videos/test.mp4"
    play_video(video_path)
