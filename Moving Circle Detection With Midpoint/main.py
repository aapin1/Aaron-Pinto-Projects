import cv2
from movingCircleDetection import detect_circles

#get video
video_path = '/Users/aaronpinto/Documents/PWP/IMG_4314.mov'

#open the video
cap = cv2.VideoCapture(video_path)

#initialize parameters
minDist = 400
param1 = 18
param2 = 10
initialMinRadius = 270
initialMaxRadius = 300

while cap.isOpened():
    #read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  #check if video is over, if it is --> break

    #call on detect_circles from movingCircleDetection.py
    frame_with_circles = detect_circles(frame, minDist, param1, param2, initialMinRadius, initialMaxRadius)

    #display the circle on the frame
    cv2.imshow('Frame', frame_with_circles)

    #check if q is pressed --> break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release vid + destroy windows
cap.release()
cv2.destroyAllWindows()
