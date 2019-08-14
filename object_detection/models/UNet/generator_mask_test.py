import os
import sys
import time
import numpy as np
import cv2
import argparse
import xml.etree.ElementTree as ET
import re
import math
import glob
import copy

def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1./scale;

def _constrain(min_v, max_v, value):
    if value < min_v: return min_v
    if value > max_v: return max_v
    return value 

def random_flip(image, flip):
    if flip == 0: return cv2.flip(image, 0)  # RIELCZ CHANGE: FLIP=0, VERTICAL FLIP
    if flip == 1: return cv2.flip(image, 1)  #                FLIP=1, HORIZONTAL FLIP
    if flip == 2: return cv2.flip(image, -1) #                FLIP=2, VERTICAL AND HORIZONTAL FLIP
    return image                             #                FLIP=3, NO FLIP

# Perform the counter clockwise rotation holding at the center
# Code adapted from: https://www.tutorialkart.com/opencv/python/opencv-python-rotate-image/
def random_rotate(image, rotation):
    if rotation == 1: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) # ROTATION=1, 90 deg CCW
    if rotation == 2: return cv2.rotate(image, cv2.ROTATE_180)                 # ROTATION=2, 180 deg
    if rotation == 3: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)        # ROTATION=3, 270 deg CCW / 90 deg CW ROTATION
    return image                                                               # ROTATION=0, NO ROTATION

def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, rotation, image_w, image_h):
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w)/image_w, float(new_h)/image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin']*sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax']*sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin']*sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax']*sy + dy))

        if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
            zero_boxes += [i]
            continue

        if (flip == 0) or (flip == 2): # VERTICAL OR BOTH FLIP
            swap = boxes[i]['ymin']
            boxes[i]['ymin'] = net_h - boxes[i]['ymax']
            boxes[i]['ymax'] = net_h - swap
        if (flip == 1) or (flip == 2): # HORIZONTAL OR BOTH FLIP
            swap = boxes[i]['xmin']
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap
        
        if rotation == 1:   # 90 deg CCW ROTATION
            swap = boxes[i]['xmin']
            boxes[i]['xmin'] = boxes[i]['ymin']
            boxes[i]['ymin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = boxes[i]['ymax']
            boxes[i]['ymax'] = net_w - swap
        elif rotation == 2: # 180 deg ROTATION
            swap1 = boxes[i]['xmin']
            swap2 = boxes[i]['ymin']
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['ymin'] = net_h - boxes[i]['ymax']
            boxes[i]['xmax'] = net_w - swap1
            boxes[i]['ymax'] = net_h - swap2
        elif rotation == 3: # 270 deg CCW / 90 deg CW ROTATION
            swap = boxes[i]['ymin']
            boxes[i]['ymin'] = boxes[i]['xmin']
            boxes[i]['xmin'] = net_h - boxes[i]['ymax']
            boxes[i]['ymax'] = boxes[i]['xmax']
            boxes[i]['xmax'] = net_h - swap
            

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes

def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5, brightness=18, quiet=True): # RIELCZ NOTE: EXPOSURE IS CONTRAST, IN THIS CONTEXT
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)
    dbrt = np.random.uniform(-brightness, brightness)
    
    if not quiet:
        print("dhue: ", dhue)
        print("dsat: ", dsat)
        print("dexp: ", dexp)
        print("dbrt: ", dbrt)

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    
    # change satuation
    image[:,:,1] *= dsat
    
    # RIELCZ CHANGE
    # change exposure (contrast) and brightness
    image[:,:,2] *= dexp
    image[:,:,2] += dbrt
    image[:,:,2] = image[:,:,2].clip(min=0., max=255.)
    
    # change hue
    image[:,:,0] += dhue
    image[:,:,0] -= (image[:,:,0] > 180)*180
    image[:,:,0] += (image[:,:,0] < 0)  *180
    
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)

def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    im_sized = cv2.resize(image, (new_w, new_h))
    
    if dx > 0: 
        im_sized = np.pad(im_sized, ((0,0), (dx,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:,-dx:,:]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0,0), (0, net_w - (new_w+dx)), (0,0)), mode='constant', constant_values=127)
               
    if dy > 0: 
        im_sized = np.pad(im_sized, ((dy,0), (0,0), (0,0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:,:,:]
        
    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h+dy)), (0,0), (0,0)), mode='constant', constant_values=127)
        
    return im_sized[:net_h, :net_w,:]     

def _aug_image(instance, net_h=416, net_w=416, quiet=True):
    image_name = instance['filename']
    image = cv2.imread(image_name) # RGB image
    
    if image is None: print('Cannot find ', image_name)
    image = image[:,:,::-1] # RGB image
        
    image_h, image_w, _ = image.shape
    
    # determine the amount of scaling and cropping
    dw = 0.2 * image_w;
    dh = 0.2 * image_h;

    new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));
    scale = np.random.uniform(0.4, 1.25); # RIELCZ CHANGE: ORIGINALLY (0.25, 2)

    if (new_ar < 1):
        new_h = int(scale * net_h);
        new_w = int(new_h * new_ar);
    else:
        new_w = int(scale * net_w);
        new_h = int(new_w * new_ar);
        
    dx = int(np.random.uniform(0, net_w - new_w));
    dy = int(np.random.uniform(0, net_h - new_h));
    
    # apply scaling and cropping
    im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
    
    # randomly distort hsv space
    im_sized = random_distort_image(im_sized, hue=18, saturation=1.25, exposure=1.25, brightness=51, quiet=quiet)
    
    # randomly flip
    flip = np.random.randint(4) # RIELCZ CHANGE
    im_sized = random_flip(im_sized, flip)
    
    # randomly rotate # RIELCZ CHANGE
    rotation = np.random.randint(4)
    im_sized = random_rotate(im_sized, rotation)
        
    # print info?
    if not quiet:
        print("dw  : ", dw)
        print("dh  : ", dh)
        print("narr: ", new_ar)
        print("scal: ", scale)
        print("neww: ", new_w)
        print("newh: ", new_h)
        print("flip: ", flip)
        print("rot : ", rotation)
    
    # correct the size and pos of bounding boxes
    all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, rotation, image_w, image_h)
    
    return im_sized, all_objs

def _gen_mask(boxes, net_h=416, net_w=416, quiet=True):
    toReturn = np.zeros((net_h, net_w, 3))
    for box in boxes:
        xmin, ymin, xmax, ymax = decodeBoxFromDict(box)
        toReturn[ymin:ymax+1, xmin:xmax+1, :] = 255
    return toReturn

def decodeBoxFromDict(box):
    '''
    Get xmin, ymin, xmax, and ymax from a box dictionary
    
    :param dict box: a dictionary defining a box containing (uint) values for the keys {x,y}{min,max}
    :return: a tuple containing (uint) xmin, ymin, xmax, ymax from the appropriate keys in box
    :rtype: tup -> int, int, int, int
    '''
    xmin = box["xmin"]
    ymin = box["ymin"]
    xmax = box["xmax"]
    ymax = box["ymax"]
    return xmin, ymin, xmax, ymax

def returnBoxDict(xmin, ymin, xmax, ymax):
    '''
    Construct a box dictionary from xmin, ymin, xmax, and ymax, with (0,0) being the top-left corner
    
    :param int xmin: the location of the min x value of the box
    :param int ymin: the location of the min y value of the box
    :param int xmax: the location of the max x value of the box
    :param int ymax: the location of the max y value of the box
    :return: a dictionary defining a box containing (uint) values for the keys {x,y}{min,max}
    :rtype: dict
    '''
    tempDict = {}
    tempDict["xmin"] = xmin
    tempDict["ymin"] = ymin
    tempDict["xmax"] = xmax
    tempDict["ymax"] = ymax
    return tempDict

def getGroundTruthBoxes(xmlfilename):
    '''
    Get a list of box dictionaries contained within a PASCAL VOC-formatted XML objects descriptor file
    
    :param str xmlfilename: the path to the XML file
    :return: a list of dictionaries defining boxes containing (uint) values for the keys {x,y}{min,max}
    :rtype: list
    '''
    if not xmlfilename:
        return None
    boxes = None
    try:
        root = ET.parse(xmlfilename)
    except FileNotFoundError:
        return None
    if root:
        boxes = []
        for objectTag in root.findall("object"):
            for bndboxTag in objectTag.findall("bndbox"):
                tempDict={}
                for child in bndboxTag:
                    tempDict[child.tag] = int(child.text)
                boxes.append(tempDict)
    return boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Data Augmentations Tester / Generator. No save option yet because it\'s not needed (yet?).')
    parser.add_argument('-i', '--input_folder', help='folder to input images', type=str, required=True)
    #parser.add_argument('-o', '--output_folder', help='folder to save modified images and labels', type=str, required=True)
    parser.add_argument('-g', '--groundtruth_folder', help='folder to PASCALVOC groundtruths.', type=str, required=True)
    parser.add_argument('-W', '--net_width', help='net width of generated image', type=int, default=416)
    parser.add_argument('-H', '--net_height', help='net height of generated image', type=int, default=416)
    parser.add_argument('--file_extension', help='the file extension of the images, including the \'.\' (default=.png)', type=str, default='.png')
    parser.add_argument('-q','--quiet', action='store_true', help='suppress console output')
    
    args = parser.parse_args()
    
    infolder  = args.input_folder
    xmlfolder = args.groundtruth_folder
    
    fileList = glob.glob(infolder + '/*' + args.file_extension)
    lenFileList = len(fileList)
    
    for filename in fileList:
        print("*** PROCESSING {} ... ***".format(filename))
        
        smallFilename = (os.path.basename(filename))
        lastDotInFilename = smallFilename.rfind('.')
        filenameBasename = smallFilename[:lastDotInFilename]
        filenameExt = smallFilename[lastDotInFilename:]

        xmlfilename = os.path.join(xmlfolder, filenameBasename + ".xml")
        groundTruths = getGroundTruthBoxes(xmlfilename)
        
        instance = {}
        instance["filename"] = filename
        instance["object"]   = groundTruths
        
        image, boxes = _aug_image(instance, net_h=args.net_height, net_w=args.net_width, quiet=args.quiet)
        for box in boxes:
            xmin, ymin, xmax, ymax = decodeBoxFromDict(box)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 105, 180), 2)
        
        mask = _gen_mask(boxes, net_h=args.net_height, net_w=args.net_width, quiet=args.quiet)
        
        cv2.imshow('original', cv2.imread(filename))
        cv2.imshow('augmented', image)
        cv2.imshow('mask', mask)
        #print(image.shape, image, mask) # DEBUG
        keyPressed = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print('-'*40)