import cv2 as cv
import numpy as np

cap = cv.VideoCapture('/home/lajith/Downloads/lane_vgt.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    blue = frame.copy()
    red = frame.copy()
    blue[:,:,1] = 0
    blue[:,:,2] = 0
    red[:,:,0] = 0
    red[:,:,1] = 0
    blue_blur = cv.GaussianBlur(blue,(5,5),0)
    red_blur = cv.GaussianBlur(red,(3,3),0)
    ret,thresh_red = cv.threshold(red_blur,127,255,cv.THRESH_BINARY)
    ret2,thresh_blue = cv.threshold(blue_blur,120,255,cv.THRESH_BINARY)
    blue_dilation = cv.dilate(thresh_blue,(5,5),iterations = 1)
    edges = cv.Canny( blue_dilation,0,200)
    #edges = np.int8(edges)
    edges = edges.astype(np.uint8)
    cv.imshow('frame', gray)
    cv.imshow('threshold_Red', thresh_red)
    cv.imshow('threshold_blue', thresh_blue)
    cv.imshow('dilation_red', edges) 

    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)

    lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    cv2.imshow("Lane",lines_edges)
    if cv.waitKey(3) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

cap.isOpened()

