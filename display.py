import cv2

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
            cv2.imshow('Frame', frame)

            # Press 'Q' on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything is done, release the VideoCapture object
    cap.release()

    # Close all the frames
    cv2.destroyAllWindows()
