import cv2
import numpy as np
import os

erodeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # TODO: Try iteratively adjusting later
closeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)) # TODO: Try iteratively adjusting later

# THIS FUNCTION IS EDITABLE! #
def thresh_mask(mask, thresh=None):
    if thresh:
        return cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)
    return cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# THIS FUNCTION IS EDITABLE! #
def postprocess_mask(mask):
    ret, th = thresh_mask(mask)
    
    eroded = cv2.erode(th, erodeKernel, iterations=1)
    opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, closeKernel, iterations=1)
    
    output = opened.copy()
    
    return ret, output

# THIS FUNCTION IS EDITABLE! #
def scale_mask(mask, image_h, image_w, net_h, net_w):
    # pass # To have no mask scaling after output 
    fx = float(image_w) / net_w
    fy = float(image_h) / net_h
    return cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
