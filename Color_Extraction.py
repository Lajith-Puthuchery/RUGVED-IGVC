import numpy as np
import cv2 as cv
cap = cv.VideoCapture('/home/lajith/Downloads/lane_vgt.mp4')


def callback(x):
    pass


cv.namedWindow('image')

ilowH = 0
ihighH = 179

ilowS = 0
ihighS = 255
ilowV = 0
ihighV = 255

# create trackbars for color change
cv.createTrackbar('lowH','image',ilowH,179,callback)
cv.createTrackbar('highH','image',ihighH,179,callback)

cv.createTrackbar('lowS','image',ilowS,255,callback)
cv.createTrackbar('highS','image',ihighS,255,callback)

cv.createTrackbar('lowV','image',ilowV,255,callback)
cv.createTrackbar('highV','image',ihighV,255,callback)
    



while(1):
    ret, frame = cap.read()
    if not ret:
    	print("Cant receive anything....")
    	break
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #cv.imshow('hsv', hsv)
    hL = cv.getTrackbarPos('H Lower','image')
    hH = cv.getTrackbarPos('H Higher','image')
    sL = cv.getTrackbarPos('S Lower','image')
    sH = cv.getTrackbarPos('S Higher','image')
    vL = cv.getTrackbarPos('V Lower','image')
    vH = cv.getTrackbarPos('V Higher','image')
    
    cv.imshow('frame',frame)
    LowerRegion = np.array([hL,sL,vL],np.uint8)
    upperRegion = np.array([hH,sH,vH],np.uint8)

    redObject = cv.inRange(hsv,LowerRegion,upperRegion)

    
    kernal = np.ones((1,1),"uint8")


    red = cv.morphologyEx(redObject,cv.MORPH_OPEN,kernal)
    red = cv.dilate(red,kernal,iterations=1)

    res1=cv.bitwise_and(frame, frame, mask = red)

    cv.imshow("Masking ",res1)

    
    if(cv.waitKey(1) & 0xFF == ord('q')):
       break


cv.destroyAllWindows()
cap.release()
