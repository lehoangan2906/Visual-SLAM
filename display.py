import cv2
from extractor.extractor import *

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

    # Downscale the frame for reducing computational loads
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))

    # Extract and Visualize features using ORB 
    # img_with_features = extract_orb_features(img)

    # Extract and Visualize features using SIFT
    # img_with_features = extract_sift_features(img)
    
    # Extract and Visualize features in each frame using AKAZE
    img_with_features = extract_akaze_features(img)

    # Extract and Visualize features in each frame using goodFeaturesToTrack
    # img_with_features = extract_good_features(img)

    # Display the processed frame
    cv2.imshow("frame", img_with_features)
    cv2.waitKey(1)
