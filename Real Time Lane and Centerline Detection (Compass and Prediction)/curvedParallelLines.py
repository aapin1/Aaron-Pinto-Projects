import cv2
import numpy as np

def process_frame(cropped_frame):
    #extract height and width of the frame
    height, width = cropped_frame.shape[:2]

    #get center coords for the rectangle (mask)
    center_x = (width // 2) + 20
    center_y = height // 2

    #rectangle dimensions
    rect_width = 400
    rect_height = 450

    #create a mask
    mask_rect = np.zeros((height, width), np.uint8)
    mask_rect = cv2.rectangle(mask_rect.copy(), (center_x - rect_width // 2, center_y - rect_height // 2),
                              (center_x + rect_width // 2, center_y + rect_height // 2), 255, -1)

    #grayscale
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    #edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    #apply the mask to only include areas inside the mask
    edges_inside_rect = cv2.bitwise_and(edges, edges, mask=mask_rect)

    #find contours
    contours, hierarchy = cv2.findContours(edges_inside_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #draw the rectangular mask
    cv2.rectangle(cropped_frame, (center_x - rect_width // 2, center_y - rect_height // 2),
                  (center_x + rect_width // 2, center_y + rect_height // 2), (255, 80, 10), 3)

    #draw contours on frame
    cv2.drawContours(cropped_frame, contours, -1, (0, 255, 0), 3)

    #if there are at least 2 contours
    if len(contours) >= 2:
        #get contour points
        points_contour1 = contours[0][:, 0, :]
        points_contour2 = contours[1][:, 0, :]

        #calculate the average between the points
        centerline_points = []
        for pt1, pt2 in zip(points_contour1, points_contour2):
            avg_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            centerline_points.append(avg_point)

        #draw the centerline by joining all different points into a line
        centerline_points = np.array(centerline_points)
        cv2.polylines(cropped_frame, [centerline_points], isClosed=False, color=(0, 0, 255), thickness=2)

    return cropped_frame
