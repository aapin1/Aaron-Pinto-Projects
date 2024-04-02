import numpy as np
import cv2
import math
import time

def region_of_interest(img, vertices):
   mask = np.zeros_like(img)
   match_mask_color = 255
   cv2.fillPoly(mask, vertices, match_mask_color)
   masked_image = cv2.bitwise_and(img, mask)
   return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
   line_img = np.zeros(
       (
           img.shape[0],
           img.shape[1],
           3
       ),
       dtype=np.uint8
   )
   img = np.copy(img)
   if lines is None:
       return
   for line in lines:
       for x1, y1, x2, y2 in line:
           cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
   img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
   return img


def pipeline(image):
   global negative_slope_count
   global positive_slope_count
   global display_duration
 
   height, width = image.shape[:2]
    # Region of interest vertices
   region_of_interest_vertices = [
       (100, height),
       (width / 2, height / 2),
       (width - 400, height),
   ]
    # Convert to grayscale
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive histogram equalization to enhance contrast
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
   enhanced_gray = clahe.apply(gray_image)
    # Gaussian blur
   blurred_image = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    # Canny edge detection with adjusted thresholds
   edges = cv2.Canny(blurred_image, 50, 150)
    # Region of interest
   masked_edges = region_of_interest(
       edges,
       np.array(
           [region_of_interest_vertices],
           np.int32
       ),
   )
    # Hough lines
   lines = cv2.HoughLinesP(
       masked_edges,
       rho=6,
       theta=np.pi / 60,
       threshold=160,
       lines=np.array([]),
       minLineLength=40,
       maxLineGap=25
   )


   # Logic to detect dashed lines and draw consolidated lines
   left_line_x = []
   left_line_y = []
   right_line_x = []
   right_line_y = []
   if lines is not None:
       for line in lines:
           for x1, y1, x2, y2 in line:
               slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
               if math.fabs(slope) < 0.5:
                   continue
               if slope <= 0:
                   left_line_x.extend([x1, x2])
                   left_line_y.extend([y1, y2])
               else:
                   right_line_x.extend([x1, x2])
                   right_line_y.extend([y1, y2])


   min_y = int(height * (3 / 5))
   max_y = height


   # Fit lines to points and draw consolidated lines
   line_image = np.zeros_like(image)
   if left_line_x and left_line_y:
       poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
       left_x_start = int(poly_left(max_y))
       left_x_end = int(poly_left(min_y))
       cv2.line(line_image, (left_x_start, max_y), (left_x_end, min_y), (0, 0, 255), thickness=25)


   if right_line_x and right_line_y:
       poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
       right_x_start = int(poly_right(max_y))
       right_x_end = int(poly_right(min_y))
       cv2.line(line_image, (right_x_start, max_y), (right_x_end, min_y), (0, 0, 255), thickness=25)


   # Draw centerline
  
   global centerline_slope


   if left_line_x and right_line_x:
       center_x_start = (left_x_start + right_x_start) // 2
       center_x_end = (left_x_end + right_x_end) // 2
       if center_x_end != center_x_start:
           centerline_slope = (min_y - max_y) / (center_x_end - center_x_start)
           cv2.line(line_image, (center_x_start, max_y), (center_x_end, min_y), (0, 255, 0), thickness=10)


  
   return cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)


# Assuming you have a video processing loop like this
cap = cv2.VideoCapture('/Users/aaronpinto/Documents/PWP/finalPWP4.mov')


# Track the count of negative slope detections for left turn prediction
negative_slope_count = 0
positive_slope_count = 0
display_duration = 0  # Variable to track the duration of the message


while cap.isOpened():
   ret, frame = cap.read()
   if ret:
       processed_frame = pipeline(frame)
      
       # Check if positive slope is detected (indicating a right turn)
       if centerline_slope > 0: 
           positive_slope_count += 1 
           display_duration = 1
           start_time = time.time() 
      
       # Check if negative slope is detected (indicating a left turn)
       elif centerline_slope < -40: 
           negative_slope_count += 1 
           display_duration = 10
           start_time = time.time() 
      
       # Display "Right turn approaching" if count is greater than 75 and the display duration is not expired
       if positive_slope_count > 90 and display_duration > 0:
           cv2.putText(processed_frame, "Right turn approaching", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
           elapsed_time = time.time() - start_time 
           if elapsed_time >= display_duration:
               positive_slope_count = 0 
               display_duration = 0 
               positive_slope_count = -1000
      
       # Display "Left turn approaching" if count is greater than 30 and the display duration is not expired
       elif negative_slope_count > 50 and display_duration > 0:
           cv2.putText(processed_frame, "Left turn approaching", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
           elapsed_time = time.time() - start_time 
           if elapsed_time >= display_duration:
               negative_slope_count = 0 
               display_duration = 0 
      
       # Display "Straight" if no turn is approaching
       else:
           cv2.putText(processed_frame, "Straight", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
      
       cv2.imshow('Processed Frame', processed_frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   else:
       break
      
cap.release()
cv2.destroyAllWindows()
