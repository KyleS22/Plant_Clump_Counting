import cv2
import numpy as np
import os

# THIS FUNCTION IS EDITABLE! #
# FOR USE IF config file's train.use_custom_mask_func == true !!! #
def custom_mask_func(boxes, net_h, net_w):
    toReturn = np.zeros((net_h, net_w, 1))
    for i in range(len(boxes)):
        thresh = 3 # originally 5
        toReturn[boxes[i]['ymin']:boxes[i]['ymax']+1, boxes[i]['xmin']:boxes[i]['xmax']+1, :] = 255
        w = (boxes[i]['xmax']-boxes[i]['xmin'])/2
        h = (boxes[i]['ymax']-boxes[i]['ymin'])/2
        cX = int(w+boxes[i]['xmin'])
        cY = int(h+boxes[i]['ymin'])
        if (w <= thresh+2) or (h <= thresh+2):
            thresh = 2 # originally 3
        if (w <= thresh+2) or (h <= thresh+2):
            thresh = 1
        toReturn[cY-thresh:cY+thresh+1, cX-thresh:cX+thresh+1, :] = 0
    return toReturn
