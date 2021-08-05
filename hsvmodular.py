
import numpy as np
import cv2 as cv
import my_module

cap = cv.VideoCapture('prac1.webm')
c = 0
kernel = np.ones((3,3), np.uint8)

while cap.isOpened():
    
    ret,frame0 = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if c == 0:
        #height and width of the frame
        height = frame0.shape[0] 
        width = frame0.shape[1]

        #creating an object of the class
        vid = my_module.video(frame0)
     

    #creating a region of image
    frame = vid.roi(frame0, 300, height, 0, width)


    #changing colorspace to HSV
    hsv = vid.color_to_HSV(frame)


    #splitting colorspace into hue, saturation and value
    h, s, v = vid.color_split(hsv)


    #applying binary threshold
    ret, thresh = vid.threshold(s, 75, 255)


    #applying median blur
    median = vid.blur(thresh, 5)


    #canny edge detedction and morphology
    edges = vid.lines_detect(median, 5, 200, kernel, 2)


    #applying hough lines to detect the lines
    lines = vid.lines_map(edges, 5, np.pi/180, 100, 20, 40)


    vid.lines_show(frame, lines)
    

    cv.imshow("frame", frame0)
    #cv.imshow("hsv", hsv)
    #cv.imshow("s", s)
    #cv.imshow("thresh", thresh)
    #cv.imshow("median", median)
    #cv.imshow("edges", edges)


    c += 1


    if cv.waitKey(1) == ord('q'):    
        break

print(c)
cap.release()
cv.destroyAllWindows()
