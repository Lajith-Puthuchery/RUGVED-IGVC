import numpy as np
import cv2 
cap = cv2.VideoCapture('/home/lajith/Downloads/lane_vgt.mp4')


class lane:

    def __init__(self,frame):
        self.frame = frame

    def gray(self,frame):
        gray =cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return gray

    def threshold(self,gray):
        ret, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        return ret, thresh

    def gaussian(self,thresh):
        g_blur = cv2.GaussianBlur(thresh,(9,9),0)
        return g_blur

    def closing(self,g_blur):
        closing = cv2.morphologyEx(g_blur, cv2.MORPH_CLOSE, kernel2)
        return closing

    def dilation(self,frame) :
        dilation = cv2.dilate(closing,kernel,iterations = 1)
        return dilation

    def erosion(self,dilation) :
        erosion = cv2.erode(dilation, kernel, iterations=1)
        return erosion

    def canny(self,erosion):
        canny = cv2.Canny(erosion,100,150)
        return canny
    
    def detect_lines(self,ROI_image,frame):

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(frame) * 0  # creating a blank to draw lines on

        lines = cv2.HoughLinesP(ROI_image, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
        return lines , line_image

    def draw_lines(self,lines, line_image):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
    

    def ROI(self,frame,vertices) :
        mask=np.zeros_like(frame)
        
        match_mask_color=(255,)
        cv2.fillPoly(mask,vertices,match_mask_color)
        masked=cv2.bitwise_and(frame,mask)
        return masked


while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    height=frame.shape[0]
    width=frame.shape[1]
    
    l = lane(frame)
    gray = l.gray(frame)
    #cv2.imshow('frame', gray)

    lower_hsv = np.array([0, 44, 126])
    higher_hsv = np.array([179, 96, 248])
    
    #ask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    #res = cv2.bitwise_and(frame,frame, mask= mask)

    kernel5 = np.ones((3,3),np.uint8)/9
    kernel = np.ones((5,5),np.uint8)/25
    kernel2 = np.ones((7,7),np.uint8)/49
    kernel3 = np.ones((5,5),np.float32)/25
    kernel4 = np.ones((11,11),np.uint8)/121

    
    ret, thresh = l.threshold(gray)
    cv2.imshow("thresh",thresh)

    g_blur = l.gaussian(thresh)
    #cv2.imshow('g_blur',g_blur)

    
    
    closing = l.closing(g_blur)
    #cv2.imshow('closing',closing)

        
    dilation = l.dilation(closing)
    #cv2.imshow('dilation',dilation)

    erosion = l.erosion(dilation)

    canny = l.canny(erosion)
    #cv2.imshow('canny',canny)

    ROI_vertices=[(0,height),(width,height),(width,height-300),(0,height-300)]

    ROI_image=l.ROI(canny, np.array([ROI_vertices]))

    lines , line_image = l.detect_lines(ROI_image, frame)
    
    l.draw_lines(lines, line_image)

    lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    cv2.imshow("Lane",lines_edges)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()