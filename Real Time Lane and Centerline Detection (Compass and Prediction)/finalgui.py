import tkinter as tk
from tkinter.ttk import *
from tkinter import *
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import math
from datetime import datetime
import time
from login import extrauser  # Importing login details

# Function to update log text display
def update_log_text():
    with open(f"{extrauser}.txt", "r") as file:
        log_content = file.read()
        log_text.config(state=tk.NORMAL)
        log_text.delete("1.0", tk.END)
        log_text.insert(tk.END, log_content)
        log_text.config(state=tk.DISABLED)
        log_text.yview(tk.END)  # Scroll to end of log text

# Function to handle movement events
def movement(direction):
    time = datetime.now()
    current_hour = time.strftime("%H")
    hourint = int(current_hour) - 5
    if hourint < 0:
        hourint = 24 + hourint
    current_time = time.strftime(f"%Y-%m-%d {hourint}:%M:%S %Z%z")

    with open(f'{extrauser}.txt', 'at') as f:
        f.write(f"{extrauser} pressed {direction} at " + current_time + "\n")

    url = "http://192.168.1.36:5000"
    
    update_log_text()

# Function to handle logout event
def logout():
    fen.destroy()
    time = datetime.now()
    current_hour = time.strftime("%H")
    hourint = int(current_hour) - 5
    if hourint < 0:
        hourint = 24 + hourint
    current_time = time.strftime(f"%Y-%m-%d {hourint}:%M:%S %Z%z")

    with open(f'{extrauser}.txt', 'at') as f:
        f.write(f"{extrauser} has logged out at " + current_time + "\n")

    update_log_text()

# Function for processing video frames
def pipeline(image):
    height, width = image.shape[:2]
    # Region of interest vertices
    region_of_interest_vertices = [
        (100, height),
        (width / 2, height / 2),
        (width - 400, height),
    ]
    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Adaptive histogram equalization to enhance contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_image)
    # Gaussian blur
    blurred_image = cv.GaussianBlur(enhanced_gray, (5, 5), 0)
    # Canny edge detection with adjusted thresholds
    edges = cv.Canny(blurred_image, 50, 150)
    # Region of interest
    masked_edges = region_of_interest(
        edges,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
    # Hough lines
    lines = cv.HoughLinesP(
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
        cv.line(line_image, (left_x_start, max_y), (left_x_end, min_y), (0, 0, 255), thickness=25)

    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        cv.line(line_image, (right_x_start, max_y), (right_x_end, min_y), (0, 0, 255), thickness=25)

    # Draw centerline
    global centerline_slope
    if left_line_x and right_line_x:
        center_x_start = (left_x_start + right_x_start) // 2
        center_x_end = (left_x_end + right_x_end) // 2
        if center_x_end != center_x_start:
            centerline_slope = (min_y - max_y) / (center_x_end - center_x_start)
            cv.line(line_image, (center_x_start, max_y), (center_x_end, min_y), (0, 255, 0), thickness=10)
    return cv.addWeighted(image, 0.8, line_image, 1.0, 0.0)

# Function to define region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

# Video capture
cap = cv.VideoCapture('/Users/aaronpinto/Documents/PWP/finalPWP4.mov')
centerline_slope = 0  # Initialize centerline slope

# Create the main Tkinter window
fen = tk.Tk()
fen.title("Processed Video")

# Define the top left frame for processed video
left = tk.Frame(fen, bg="grey", width=300, height=300)
left.pack_propagate(True)
tk.Label(left, text="Processed Video", fg="white", bg="black").pack()
left.grid(column=0, row=0, pady=5, padx=10, sticky="n")

# Create label to display video feed
video_label = tk.Label(left)
video_label.pack()

# Define the function for updating video feed
def update_video_feed():
    ret, frame = cap.read()
    if ret:
        processed_frame = pipeline(frame)
        processed_frame = cv.resize(processed_frame, (processed_frame.shape[1] // 2, processed_frame.shape[0] // 2))
        processed_frame_rgb = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
        processed_frame_pil = Image.fromarray(processed_frame_rgb)
        processed_frame_tk = ImageTk.PhotoImage(image=processed_frame_pil)
        video_label.config(image=processed_frame_tk)
        video_label.image = processed_frame_tk
    fen.after(10, update_video_feed)

update_video_feed()  # Call update_video_feed before starting the main loop

# Define the top right frame for movement controls
right = tk.Frame(fen, width=200, height=200, bg="grey")
right.pack_propagate(False)
tk.Label(right, text="Movement Controls", fg="white", bg="black").pack()
right.grid(column=2, row=0, pady=5, padx=10, sticky="n")

# Define the bottom left frame for raw video
bleft = tk.Frame(fen, bg="grey", width=200, height=200)
bleft.pack_propagate(False)
tk.Label(bleft, text="Raw Video", fg="white", bg="black", anchor="center", justify="center").pack()
bleft.grid(column=0, row=1, pady=5, padx=10, sticky="n")

# Define label widget to display raw video
label_widget = Label(bleft)
label_widget.pack()

# Function to open camera and display raw video
def open_cam():
    _, frame = cap.read()
    opencv_image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, open_cam)

open_cam()

# Define the bottom right frame for log
bright = tk.Frame(fen, bg="grey", width=200, height=200)
bright.pack_propagate(False)
tk.Label(bright, text="Log", fg="white", bg="black").pack()
log_text = tk.Text(bright, fg="white", bg="black", height=10, width=30, wrap=tk.WORD, state=tk.DISABLED)
log_text.grid(column=0, row=2, pady=22, padx=10, sticky="n", rowspan=2)

# Define scrollbar for log text
scrollbar = tk.Scrollbar(bright, command=log_text.yview)
log_text.config(yscrollcommand=scrollbar.set)

bright.grid(column=2, row=1, pady=5, padx=10, sticky="n")

fen.mainloop()  # Start the main event loop
