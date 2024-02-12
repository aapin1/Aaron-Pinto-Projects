import cv2
import numpy as np

def angle_between_lines(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    angle_diff = np.abs(theta1 - theta2)
    return angle_diff

def line(frame, lines):
    # Filter out parallel lines and draw the centerline
    if lines is not None:
        parallel_lines = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                angle_diff = angle_between_lines(line1, line2)
                
                # Set the angle threshold for considering lines as parallel
                angle_threshold = np.pi / 18  # Approximately 10 degrees
                
                if angle_diff < angle_threshold:
                    parallel_lines.append(line1)
                    parallel_lines.append(line2)
        
        # Check if there are parallel lines
        if parallel_lines:
            # Calculate the mean rho and theta values for the parallel lines
            rho_values = np.array([line[0] for line in parallel_lines])
            theta_values = np.array([line[1] for line in parallel_lines])
            mean_rho = np.mean(rho_values)
            mean_theta = np.mean(theta_values)
            
            # Draw the centerline
            a = np.cos(mean_theta)
            b = np.sin(mean_theta)
            x0 = a * mean_rho
            y0 = b * mean_rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  

        # Draw the parallel lines
        for rho, theta in parallel_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
