import cv2
from curvedParallelLines import process_frame

def main():
    #initialize vid capture
    vid = cv2.VideoCapture(0)

    #loop through frames
    while True:
        #read one frame
        ret, frame = vid.read()

        #if video ends --> exit
        if not ret:
            break

        #send to other file to process
        processed_frame = process_frame(frame)

        #display processed file
        cv2.imshow('Contours', processed_frame)

        #if ESC pressed --> exit
        if cv2.waitKey(1) == 27:
            break

    #release & destroy
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
