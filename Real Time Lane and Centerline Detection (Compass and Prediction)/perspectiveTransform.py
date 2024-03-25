# import necessary libraries
import cv2
import numpy as np

# Turn on Laptop's webcam
cap = cv2.VideoCapture("IMG_5199.MOV")

while True:

    ret, frame = cap.read()

    # Locate points of the documentsy
    # or object which you want to transform
    pts1 = np.float32([[-1050, 860], [-430, 860],
                       [-1050, 1000], [-430, 1000]])
    pts2 = np.float32([[-1050, 600], [-650, 600],
                       [-1050, 1240], [-650, 1240]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (500, 600))

    # Wrap the transformed image
    cv2.imshow('frame', frame)  # Initial Capture
    cv2.imshow('frame1', result)  # Transformed Capture

    if cv2.waitKey(24) == 27:
        break

cap.release()
cv2.destroyAllWindows()
