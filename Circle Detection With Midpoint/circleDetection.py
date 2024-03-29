import numpy as np
import cv2

img = cv2.imread('soda.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.medianBlur(gray, 25) 

minDist = 100
param1 = 30 
param2 = 150
minRadius = 880
maxRadius = 1200 

circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if circles is not None:
   circles = np.uint16(np.around(circles))
   for i in circles[0,:]:
       cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 16)
   for pt in circles[0, :]:
       a,b,r = pt[0], pt[1], pt[2]
       cv2.circle(img, (a,b),1,(0,0,255),30)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

