# import necessary libraries
import cv2
import numpy as np

# Turn on Laptop's webcam
cap = cv2.VideoCapture("IMG_5199.MOV")

while True:

    ret, frame = cap.read()

    # Locate points of the documentsy
    # or object which you want to transform
    pts1 = np.float32([[-1050, 960], [-430, 960],
                       [-1050, 1100], [-430, 1100]])
    pts2 = np.float32([[-1050, 700], [-650, 700],
                       [-1050, 1340], [-650, 1340]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (500, 600))

    # frame is 1080 by 1920
    cropped_image = frame[0:900, 0:1920]

    #height, width = frame.shape[:2]
    #print(f"height: {height}")
    #print(f"width: {width}")


    # Wrap the transformed image
    cv2.imshow('frame', cropped_image)  # Initial Capture
    cv2.imshow('frame1', result)  # Transformed Capture

    if cv2.waitKey(24) == 27:
        break

cap.release()
cv2.destroyAllWindows()
