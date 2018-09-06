import cv2
import numpy as np
import matplotlib.pyplot as plt
def canny(image):
    #convert color to gray image
    """
    Edge Detection technique : Identify sharp change in intnsity in adjacent pixels
    Image is an array of pixels
    Step1: Convert colored image to Gray Scale image
    """
    grey=cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    """
    Step2 : Reduce Noise using Gaussian Blur
    """
    blur = cv2.GaussianBlur(grey,(5,5),0)
    """
    Step3: Apply Canny Edge Detection
    derivative(f(x,y))
    """
    canny_edge=cv2.Canny(blur,50,150)
    return canny_edge

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            #unpack
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0,15))
    return line_image
"""
Step 4 Region of Interest
"""
def region_of_interest(image):
    height=image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height) ,(550,250)]
    ])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    """
    Step 5 Bitwise AND
    """
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image
image =cv2.imread('test_image.jpg')
lane_image=np.copy(image)
canny_edge=canny(lane_image)
cropped_image=region_of_interest(canny_edge)
"""
Step 6 : Hough Transform
"""
lines=cv2.HoughLinesP(cropped_image, 2, (np.pi/180) ,100, np.array([]), minLineLength=40, maxLineGap=5)
line_image=display_lines(lane_image,lines)
combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,0)
cv2.imshow("result",combo_image)
cv2.waitKey(0)
