import cv2
import numpy as np

def detect_circles(frame, minDist, param1, param2, initialMinRadius, initialMaxRadius):
    #convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #apply median blur
    blurred = cv2.medianBlur(gray, 25)

    #set parameters
    minRadius = initialMinRadius
    maxRadius = initialMaxRadius

    #use houghcircles for detection
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None: #if circles are detected
        circles = np.uint16(np.around(circles)) 
        i = circles[0, 0] #gets first circle

        #circle
        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 30)

        #midpoint
        cv2.circle(frame, (i[0], i[1]), 1, (0, 0, 255), 30)

    return frame

