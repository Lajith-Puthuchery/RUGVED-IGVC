
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

    #blur = cv.GaussianBlur(s, (5,5), 0)

    lower_green= np.array([0])
    upper_green = np.array([80]) 
    #mask = cv.inRange(s, lower_green, upper_green)
    #mask = cv.bitwise_not(mask)
    #res = cv.bitwise_and(s, s, mask = mask)

    #cv.imshow("h", h)

    ret, thresh = cv.threshold(h, 50, 255, cv.THRESH_BINARY)
    #med2 = cv.medianBlur(thresh2, 3)
    
    ret, thresh2 = cv.threshold(s, 70, 255, cv.THRESH_BINARY_INV)

    

    #th = cv.adaptiveThreshold(s,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15,20)


    #blur = cv.GaussianBlur(thresh, (5,5), 0)
    #blur = cv.medianBlur(thresh,5)
    #blur = cv.bilateralFilter(thresh, 10, 150, 150)


    #morph = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel1)

    #morph = cv.dilate(opening, kernel1, iterations = 1)
    #print(med2)

    fin = (thresh / 2) + (thresh2 / 2)
    #print(fin)
    ret, thresh3 = cv.threshold(fin, 128, 255, cv.THRESH_BINARY)
    thresh3 = thresh3.astype(np.uint8)
    #cv.imshow("fin", fin)
    #cv.imshow("thresh3", thresh3)
    #print(thresh.dtype)

    #morph3 = cv.dilate(thresh3, kernel2, iterations = 1)
    #cv.imshow("morph3", morph3)

    edges = cv.Canny(thresh3, 5, 200)


    morph2 = cv.dilate(edges, kernel1, iterations = 1)
    median = cv.medianBlur(morph2, 7)

    #opening = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel1)


    


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
