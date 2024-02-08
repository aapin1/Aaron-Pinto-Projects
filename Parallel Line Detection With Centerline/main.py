import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    # create a mask
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[100:1000, 300:850] = 255

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY).

    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

    lines_list = []
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=5, maxLineGap=10)

    for points in lines:
        x1, y1, x2, y2 = points[0]
        cv2.line(masked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines_list.append([(x1, y1), (x2, y2)])

    cv2.imshow('Line Detection', masked_img)


    if cv2.waitKey(1) == 27:
        exit(0)

#make mask follow around the paper
#use line detectipon in masked image

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
