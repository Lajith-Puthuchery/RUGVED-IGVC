import cv2 as cv

class video:

    def __init__(self, frame):
        self.frame = frame

        
    def roi(self, frame, row1, row2, col1, col2):
        frame = frame[row1 : row2, col1 : col2]
        return frame


    def color_to_HSV(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        return hsv


    def color_split(self, frame):
        h, s, v = cv.split(frame)
        return h, s, v


    def threshold(self, frame, threshval, maxval):
        ret, thresh = cv.threshold(frame, threshval, maxval, cv.THRESH_BINARY_INV)
        return ret, thresh


    def blur(self, frame, kernel_size):
        median = cv.medianBlur(frame, kernel_size)
        return median


    def lines_detect(self, frame, canny_minval, canny_maxval, kernel, num_iterations):
        canny = cv.Canny(frame, canny_minval, canny_maxval)
        morph = cv.dilate(canny, kernel, iterations = num_iterations)
        return morph


    def lines_map(self, frame, rho, theeta, threshold, minlinelength, maxlinegap):
        lines = cv.HoughLinesP(frame, rho, theeta, threshold, minLineLength = minlinelength, maxLineGap = maxlinegap)
        return lines


    def lines_show(self, frame, lines):
        for line in lines:
            x1,y1,x2,y2 = line[0] 
            cv.line(frame,(x1,y1),(x2,y2),(255,0,255),5)



    

