import cv2
from extractor.extractor import *


# Global variables to store the previous frame and its state
prev_img = None
prev_kp = None
prev_good_matches = None


# For displaying the video
def play_video(video_path):
    # Create VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_path)

    # Check if camera is opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Read until the video is completed
    while cap.isOpened():
        # Capture frame by frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            #cv2.imshow('Frame', frame)

            # Press 'Q' on keyboard to exit
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    break
            
            # Downscale the frame for reducing computational loads
            frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))

            # Processing each frame and display them
            process_frame(frame)
        else:
            break

    # When everything is done, release the VideoCapture object
    cap.release()

    # Close all the frames
    cv2.destroyAllWindows()


# For processing each frame
def process_frame(img):
    global prev_img, prev_kp, prev_good_matches

    # Extract and Visualize features using ORB 
    # img_with_features = extract_orb_features(img)

    # Extract and Visualize features using SIFT
    # img_with_features = extract_sift_features(img)

    # Extract and Visualize features in each frame using goodFeaturesToTrack
    # img_with_features = extract_good_features(img)


    # Initialize on the first frame
    if prev_img is None:
        prev_img = img
        cv2.imshow("frame", prev_img)
        cv2.waitKey(1)
        return

    
    # Extract keypoints and matches for the current and previous frame
    kp1, kp2, good_matches = extract_akaze_features(prev_img, img)

    # Draw the matches between two frames
    if good_matches and len(kp1) > 0 and len(kp2) > 0:
        img_matches = cv2.drawMatches(prev_img, kp1, img, kp2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("frame", img_matches)
    else:
        cv2.imshow("frame", img)    # Fallback if no matches

    
    # Update previous frame for the next iteration
    prev_img = img
    prev_kp = kp2
    prev_good_matches = good_matches

    cv2.waitKey(1)
