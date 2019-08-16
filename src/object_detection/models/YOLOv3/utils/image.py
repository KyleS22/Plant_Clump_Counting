import cv2
import numpy as np
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

def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5, brightness=18): # RIELCZ NOTE: EXPOSURE IS CONTRAST, IN THIS CONTEXT
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)
    dbrt = np.random.uniform(-brightness, brightness)

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