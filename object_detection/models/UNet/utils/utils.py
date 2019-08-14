import cv2
import numpy as np
import os
from .bbox import BoundBox, bbox_iou
from .mask_postprocess import thresh_mask, postprocess_mask, scale_mask
from scipy.special import expit

def _sigmoid(x):
    return expit(x)

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def get_instances_from_bbox(boxes):
    '''
    Args:
        boxes: a list of class BoundBox from Experiencor

    Returns:
        list of dictionaries with fields ymin, xmin, ymax, xmax, class, score
    '''
    box_map = []
    for bbox in boxes:
        tempDict = {}
        tempDict["ymin"] = bbox.ymin
        tempDict["xmin"] = bbox.xmin
        tempDict["ymax"] = bbox.ymax
        tempDict["xmax"] = bbox.xmax
        box_map.append(tempDict)
    return box_map

def evaluate(model, 
             generator, 
             iou_threshold=0.5,
             nms_thresh=0.45,
             net_h=416,
             net_w=416,
             save_path=None,
             save_format='.png',
             quiet=True):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet

    # Arguments
        model           : The model to evaluate.
        generator       : The generator that represents the dataset to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        nms_thresh      : The threshold used to determine whether two detections are duplicates
        net_h           : The height of the input image to the model, higher value results in better accuracy
        net_w           : The width of the input image to the model
        save_path       : The path to save images with visualized detections to.
        save_format     : The format in which to save images.
        quiet           : If true, shows each bounded image and mask.
    # Returns
        A dict mapping class names to mAP scores.
    """    
    # gather all detections and annotations
    all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    
    for i in range(generator.size()):
        raw_image = [generator.load_image(i)]
        raw_fname = [generator.load_filename(i)]

        # make the boxes and the labels
        pred_boxes_raw, ret_raw, mask_raw = get_unet_boxes(model, raw_image, net_h, net_w, nms_thresh, return_mask=True)
        pred_boxes = pred_boxes_raw[0]
        ret        = ret_raw[0]
        pred_mask  = mask_raw[0]
        
        #score = np.array([box.get_score() for box in pred_boxes])
        #pred_labels = np.array([box.label for box in pred_boxes]) # ORIGINAL
        pred_labels = np.array([generator.num_classes()-1] * len(pred_boxes)) # Due to binary nature of Unet
        
        if len(pred_boxes) > 0:
            box_inst = get_instances_from_bbox(pred_boxes)
            pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes]) 
            #print(pred_boxes) # DEBUG
        else:
            pred_boxes = np.array([[]])  
            box_inst   = []
        
        if save_path:
            for image, filename in zip(raw_image, raw_fname):
                temp = image.copy()
                #for box in pred_boxes:
                #    cv2.rectangle(temp, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 3)
                for box in box_inst:
                    cv2.rectangle(temp, (box["xmin"],box["ymin"]), (box["xmax"],box["ymax"]), (0,255,0), 3)
                if not quiet:
                    cv2.imshow(filename, temp)
                    cv2.imshow('true_mask', true_mask)
                    cv2.imshow('pred_mask', pred_mask)
                    keyPressed = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_path, os.path.basename(filename[:filename.rfind('.')])+'_mask'+save_format), pred_mask)
                cv2.imwrite(os.path.join(save_path, os.path.basename(filename[:filename.rfind('.')])+save_format), temp)
        # sort the boxes and the labels according to scores
        #score_sort = np.argsort(-score)
        #pred_labels = pred_labels[score_sort]
        #pred_boxes  = pred_boxes[score_sort]
        
        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]
        
        annotations = generator.load_annotation(i)
        
        # copy annotations to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
    
    #for label in range(generator.num_classes()):
    #    print(all_detections[0][label]) # DEBUG
    #    print(all_annotations[0][label]) # DEBUG
    
    # compute mAP (and other things) by comparing all detections and all annotations
    average_precisions = {}
    
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        #scores          = np.zeros((0,))
        num_annotations = 0.0
        
        #rets     = np.zeros((0,))
        #jaccard  = np.zeros((0,))
        #dice     = np.zeros((0,))
        
        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                #scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                
        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        #indices         = np.argsort(-scores)
        #false_positives = false_positives[indices]
        #true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = compute_ap(recall, precision)  
        average_precisions[label] = average_precision
        
        # Compute other averages
        #average_rets[label] = rets
    
    return average_precisions

def correct_unet_boxes(boxes, image_h, image_w, net_h, net_w):
    '''if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h'''
        
    x_scale = float(image_w)/net_w
    y_scale = float(image_h)/net_h
    
    for i in range(len(boxes)):
        boxes[i].xmin = int(boxes[i].xmin * x_scale)
        boxes[i].xmax = int(boxes[i].xmax * x_scale)
        boxes[i].ymin = int(boxes[i].ymin * y_scale)
        boxes[i].ymax = int(boxes[i].ymax * y_scale)
        
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

def normalize(image):
    return image/255.

def unnormalize(image):
    image *= 255.
    return image.astype('uint8')

def get_unet_boxes(model, images, net_h, net_w, nms_thresh, return_mask=False):
    image_h, image_w, _ = images[0].shape
    nb_images           = len(images)
    batch_input         = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)        

    # run the prediction
    batch_output = model.predict_on_batch(batch_input)
    batch_boxes  = [None]*nb_images
    batch_rets   = [None]*nb_images
    batch_masks  = [None]*nb_images

    for i in range(nb_images):
        mask = batch_output[i]
        mask = unnormalize(mask)
        ret, mask = postprocess_mask(mask)
        boxes = []

        # decode the mask
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append(BoundBox(x, y, x+w, y+h, -1, [0]))

        # correct the sizes of the bounding boxes
        correct_unet_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)        
        
        mask = scale_mask(mask, image_h, image_w, net_h, net_w)
        
        batch_boxes[i] = boxes
        batch_rets[i]  = ret
        batch_masks[i] = mask
    
    if return_mask:
        return batch_boxes, batch_rets, batch_masks
    return batch_boxes

def compute_jaccard(a, b):
    a_ = normalize(a).astype('bool')
    b_ = normalize(b).astype('bool')
    intersection = np.logical_and(a_, b_)
    union = np.logical_or(a_, b_)
    return np.sum(intersection) / np.sum(union)

def compute_dice(a, b):
    a_ = normalize(a).astype('bool')
    b_ = normalize(b).astype('bool')
    intersection = np.logical_and(a_, b_)
    return 2 * np.sum(intersection) / (np.sum(a_) + np.sum(b_))

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap     

def _softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)
