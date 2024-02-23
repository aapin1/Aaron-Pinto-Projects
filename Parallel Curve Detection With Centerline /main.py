import cv2
import numpy as np

# Initialize video capture
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    if not ret:
        break  # Exit the loop if frame is None or there's an error

    # Resize frame to fit display window
    frame = cv2.resize(frame, (800, 600))

    height, width = frame.shape[:2]

    # Center coordinates for the rectangle
    center_x = width // 2
    center_y = height // 2

    # New rectangle size (make it bigger)
    rect_width = 250
    rect_height = 400

    # Create a mask for the rectangle area
    mask_rect = np.zeros((height, width), np.uint8)
    mask_rect = cv2.rectangle(mask_rect.copy(), (center_x - rect_width // 2, center_y - rect_height // 2),
                              (center_x + rect_width // 2, center_y + rect_height // 2), 255, -1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply the mask to include the rectangle area in edge detection
    edges_inside_rect = cv2.bitwise_and(edges, edges, mask=mask_rect)

    contours, hierarchy = cv2.findContours(edges_inside_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw the rectangle
    cv2.rectangle(frame, (center_x - rect_width // 2, center_y - rect_height // 2),
                  (center_x + rect_width // 2, center_y + rect_height // 2), (255, 80, 10), 3)

    for eachContour in contours:
        pass


    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Contours', frame)

    # Check for key press to exit
    if cv2.waitKey(1) == 27:
        break




