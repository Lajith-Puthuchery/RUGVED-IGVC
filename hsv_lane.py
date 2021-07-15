
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('lane_vgt.mp4')
c = 0

#fourcc = cv.VideoWriter_fourcc(*'MJPG')
#out = cv.VideoWriter('lines_detection.avi', fourcc, 60.0, (640,  480))


while cap.isOpened():
    
    ret,frame0 = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = frame0[200: , :]

    kernel1 = np.ones((5,5), np.uint8)
    #kernel2 = np.ones((5,5), np.uint8)

    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv)


    ret, thresh = cv.threshold(h, 50, 255, cv.THRESH_BINARY)
    #med2 = cv.medianBlur(thresh2, 3)
    
    ret, thresh2 = cv.threshold(s, 70, 255, cv.THRESH_BINARY_INV)


    fin = (thresh / 2) + (thresh2 / 2)
    ret, thresh3 = cv.threshold(fin, 128, 255, cv.THRESH_BINARY)
    thresh3 = thresh3.astype(np.uint8)



    edges = cv.Canny(thresh3, 5, 200)


    morph2 = cv.dilate(edges, kernel1, iterations = 1)
    median = cv.medianBlur(morph2, 7)




    


    lines = cv.HoughLinesP(median, 5, np.pi/180, 100, minLineLength = 10, maxLineGap = 30)
    for line in lines:
         x1,y1,x2,y2 = line[0]
         
         cv.line(frame,(x1,y1),(x2,y2),(255,0,255),3, lineType = cv.FILLED)
    
    cv.imshow("frame", frame0)
    #cv.imshow("hsv", hsv)
    #cv.imshow("s", s)
    #cv.imshow("Split HSV",hsv_split)
    #cv.imshow("thresh", thresh)
    #cv.imshow("thresh2,", thresh2)
    #cv.imshow("med2", med2)
    #cv.imshow("th", th)
    #cv.imshow("reds", res)
    #cv.imshow("blur", blur)
    #cv.imshow("morph", morph)
    #cv.imshow("edges", edges)
    #cv.imshow("open", morph)
    #cv.imshow("morph2", morph2)
    cv.imshow("med", median)
    #cv.imshow("frame original", frame0)
    #cv.imshow("blur3", blur3)


    
    c += 1
    #out.write(frame)

    if cv.waitKey(1) == ord('q'):    
        break
print(c)
cap.release()
cv.destroyAllWindows()
