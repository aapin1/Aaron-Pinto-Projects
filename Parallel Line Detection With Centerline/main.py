import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    # create a mask
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[100:350, 150:550] = 255

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Masked Image', masked_img)

    if cv2.waitKey(1) == 27:
        exit(0)

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
