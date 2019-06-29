'''
MODIFIED FROM CODE WRITTEN BY RIEL CASTRO-ZUNTI
FOR THE STRAWBERRY DETEFCTION PROJECT FOR CMPY 819
'''

# ***** IMPORTS START HERE ***** #

import os
import sys
import time
import numpy as np
import cv2
import argparse
import xml.etree.ElementTree as ET
import re
import math

import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

# ***** METHODS START HERE ***** #

def get_boxes_from_DL(boxes, classes, scores, category_index, min_score_thresh=0.5):
    '''
    Args:
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
            and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then
            this function assumes that the boxes to be plotted are groundtruth
            boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
            category index `id` and category name `name`) keyed by category indices.
        min_score_thresh: minimum score threshold for a box to be recognized.

    Returns:
        list of dictionaries with fields ymin, xmin, ymax, xmax, class, score
    '''
    box_map = []
    for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh: # Pass confidence check
            box = tuple(boxes[i].tolist())
            tempDict = {}
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = "N/A"
            score = 100*scores[i]
            tempDict["ymin"] = box[0]
            tempDict["xmin"] = box[1]
            tempDict["ymax"] = box[2]
            tempDict["xmax"] = box[3]
            tempDict["class"] = class_name
            tempDict["score"] = score
            box_map.append(tempDict)
    return box_map

def unpolar_boxes(box_map, img):
    '''
    Convert bounding box coordinates from polar to absolute uint8
    
    :param list box_map: a list containing dictionarie that defining a prediction box containing (float) values for the keys {x,y}{min,max}
    :param array img: a numpy array representing an image with a width and height
    :after: all dictionaries in box_map have the values associated with the keys x{min,max} multiplied by the img width, and the values associated with the keys y{min,max} multiplied by the img height
    '''
    h, w, _ = img.shape
    for i in range(len(box_map)):
        box_map[i]["ymin"] *= h
        box_map[i]["xmin"] *= w
        box_map[i]["ymax"] *= h
        box_map[i]["xmax"] *= w
        box_map[i]["ymin"] = int(math.floor(box_map[i]["ymin"]))
        box_map[i]["xmin"] = int(math.floor(box_map[i]["xmin"]))
        box_map[i]["ymax"] = int(math.ceil(box_map[i]["ymax"]))
        box_map[i]["xmax"] = int(math.ceil(box_map[i]["xmax"]))

def closestGroundTruthBox(testBox, groundTruthBoxes):
    '''
    Find the closest (based on minimum total Manhattan Distance)
    prediction box to a ground-truth box given a list of 
    ground-truth boxes
    
    :param dict testBox: a dictionary defining a prediction box containing (uint) values for the keys {x,y}{min,max}
    :param list groundTruthBoxes: a list of dictionaries defining ground-truth boxes containing (uint) values for the keys {x,y}{min,max}
    :return: the box in groundTruthBoxes with the minimum sum of Manhattan distance differences (over {x,y}{min,max}) compared to testBox
    :rtype: dict
    '''
    # Test for if there's only one box
    if len(groundTruthBoxes) == 1:
        return groundTruthBoxes[0]
    # Multiple boxes, find min sum of Euclidean distance
    bestbox = None
    minTotalDist = np.inf
    testBoxA = np.array((testBox["xmin"], testBox["ymin"]))
    testBoxB = np.array((testBox["xmax"], testBox["ymax"]))
    for box in groundTruthBoxes:
        currGTA = np.array((box["xmin"], box["ymin"]))
        currGTB = np.array((box["xmax"], box["ymax"]))
        currTotalDist = np.abs(np.linalg.norm(testBoxA-currGTA)) + np.abs(np.linalg.norm(testBoxB-currGTB))
        if currTotalDist < minTotalDist:
            bestbox = box
            minTotalDist = currTotalDist
    return bestbox

def bb_intersection_over_union(boxA, boxB):
    '''
    Find the Intersection over Union (Jaccard or Tanimoto) of two bounding boxes
    
    Modified from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    
    :param dict boxA: a dictionary defining a box containing (uint) values for the keys {x,y}{min,max}
    :param dict boxB: a dictionary defining a box containing (uint) values for the keys {x,y}{min,max}
    :return: the Intersection over Union between boxA and boxB
    :rtype: float
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA["xmin"], boxB["xmin"])
    yA = max(boxA["ymin"], boxB["ymin"])
    xB = min(boxA["xmax"], boxB["xmax"])
    yB = min(boxA["ymax"], boxB["ymax"])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA["xmax"] - boxA["xmin"] + 1) * (boxA["ymax"] - boxA["ymin"] + 1)
    boxBArea = (boxB["xmax"] - boxB["xmin"] + 1) * (boxB["ymax"] - boxB["ymin"] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

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

def getMaxIDfromPbtxt(pbtxtfilename):
    listOfIDs = []
    with open(pbtxtfilename, 'r') as f:
        for line in f:
            if re.match('^\s*id:\s*[0-9]+\s*$', line):
                listOfIDs.extend([int(s) for s in line.split() if s.isdigit()])
    return max(listOfIDs)


# ***** MAIN STARTS HERE ***** #

## *** INPUT ARGS *** ##
parser = argparse.ArgumentParser(description='Process a series of images using a Tensorflow Object Detection API exported inference graph (*.pb).')
parser.add_argument('ckpt', metavar='C', nargs=1,
                    help='the path/filename to the exported inference graph (a PB file)')
parser.add_argument('labels', metavar='L', nargs=1,
                    help='the path/filename to the labels PBTXT file')
parser.add_argument('-i', '--infolder', nargs=1, dest='input_folder',
                    default='.',
                    help='the input folder (current folder is default)')
parser.add_argument('-o', '--outfolder', nargs=1, dest='output_folder',
                    default='.',
                    help='the destination folder (current folder is default)')
parser.add_argument('-g', '--ground-truth-folder', nargs=1, dest='xmlfilenames',
                    default=None,
                    help='the location of the xml files referring to the ground-truth bounding box')
parser.add_argument('-a', '--area-percent', dest='bbox_alpha',
                    type=float, default=None,
                    help='a bounding box should have a minimum of this (float) ratio bounding box area to image size to be recognized (default is no minimum)')
parser.add_argument('-c', '--confidence', dest='bbox_confidence',
                    type=float, default=0.5,
                    help='a detected bounding box should have a minimum of this (float between 0 and 1) confidence to be considered recognized by the deep learning inference process (0.5 [50%%] is default)')
parser.add_argument('-j', '--iou', dest='iou_thresh_min',
                    type=float, default=0.5,
                    help='minimum Jaccard (Tanimoto) index; a detected bounding box should have a minimum of this (float between 0 and 1) Intersection over Union (IoU) against its closest ground truth bounding box to be considered recognized and thus considered within aggregate recognized IoU measurements (0.5 [50%%] is default)')
parser.add_argument('-w', '--max-width', dest='max_width',
                    type=int, default=500,
                    help='the maximum width for an input image; image is scaled such that its width is this if its width is greater than this; default is 500, but entering a negative value means no resizing')
parser.add_argument('-q','--quiet', action='store_true',
                    help='show no OpenCV windows')
parser.add_argument('-s', '--save-stats', action='store_true',
                    help='outputs statistics to a file whose name is <filename>.txt in addition to stdout')
parser.add_argument('--save-in-new-folder', action='store_true',
                    help='save all output to a subfolder in output_folder whose name is <filename>')
parser.add_argument('--save-aggregate-statistics', action='store_true',
                    help='save the aggregate statistics in a .CSV file in output_folder')
parser.add_argument('--save-aggregate-statistics-readout', action='store_true',
                    help='save the aggregate statistics readout (printed to stdout at end) in a .TXT file in output_folder')
parser.add_argument('--save-aggregate-statistics-filename', nargs=1, dest='save_aggregate_statistics_filename',
                    default='aggregateGlobalStatsDL',
                    help='the filename of the aggregate statistics .CSV / .TXT file, if either save-aggregate-statistics or save-aggregate-statistics-readout is set; should not include the \".csv\" or \".txt\" file extension')
parser.add_argument('-n', '--dry-run', action='store_true',
                    help='don\'t save anything')


## *** PARSE AND DEAL WITH INPUTS *** ##

args = parser.parse_args()

NUM_CLASSES = getMaxIDfromPbtxt(args.labels[0])

PATH_TO_CKPT = os.path.join(args.ckpt[0])

infolder = args.input_folder[0]

outfolder = args.output_folder[0]

if args.xmlfilenames:
    xmlfolder = args.xmlfilenames[0]
else:
    xmlfolder = None

MAXWIDTH = args.max_width
MAXVAL = 255

BBOX_ALPHA = args.bbox_alpha # 10% of image BBOX area is bounding box
BBOX_CONFIDENCE = args.bbox_confidence # 0.5 # % of image area is bounding box
IOU_THRESH_MIN = args.iou_thresh_min

### DEAL WITH CREATING AGGREGATE PER-IMAGE CSV, IF REQUESTED ###
globalStatsFilename = args.save_aggregate_statistics_filename[0]
globalStatsCSVFilename = globalStatsFilename + '.csv'
globalStatsTXTFilename = globalStatsFilename + '.txt'
if args.save_aggregate_statistics:
    with open(os.path.join(outfolder, globalStatsCSVFilename), 'w+') as f:
        f.write("image,time0,time1,time2,time3,numboxesfound,numboxesGT,avIoUtotal,avIoUrec,RR,DR,UR,MR,precision,recall,fscore\n")

## *** FETCH INPUT IMAGES *** ##
infiles = []
for f in os.listdir(infolder):
    temppath = os.path.join(infolder, f)
    if os.path.isfile(temppath):
        infiles.append(temppath)

## *** PREPARE AGGREGATE LISTS *** ##

globalIOUsRecognized = []
globalIOUsTotal = []

globalTimeWithoutPictureImportAndResizingTime = []
globalTimeTotal = []

globalRR = []
globalDR = []
globalMR = []

globalPrecision = []
globalRecall = []
globalFscore = []

## *** OTHER STARTUP INITIALIZATIONS *** #

groundTruths = None # To be changed per-image if GT folder specified
hasProcessedFirst = False # Set to true after first inference

### LOAD A (FROZEN) TENSORFLOW MODEL INTO MEMORY ###

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

### LOAD LABEL MAP ###
# Label maps map indices to category names.
label_map = label_map_util.load_labelmap(args.labels[0])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

## *** DETECTION DRIVER START *** ##
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for filename in infiles:
            #### Import Image ####
            smallFilename = (os.path.basename(filename))
            lastDotInFilename = smallFilename.rfind('.')
            filenameBasename = smallFilename[:lastDotInFilename]
            filenameExt = smallFilename[lastDotInFilename:]

            #### Fetch Ground-truth Boxes ####
            if xmlfolder:
                xmlfilename = os.path.join(xmlfolder, filenameBasename + ".xml")
                groundTruths = getGroundTruthBoxes(xmlfilename)

            ## *** IMREAD STARTS HERE *** ##
            print("*** PROCESSING {} ... ***".format(smallFilename))
            start_time = time.clock()
            frame = cv2.imread(filename)
            if frame is None: 
                print("Warning: {} could not be read. Skipping file...".format(filename))
                continue
            time0 = time.clock() - start_time
            print("--- IMAGE READ COMPLETE: %s seconds ---" % time0)

            ### INOUT CHECKS ###
            
            height, width, channels = frame.shape

            ## *** IMAGE RESIZING STARTS HERE *** ##
            imageResized = False # Assume false unless proven true
            if (MAXWIDTH > 0) and (width > MAXWIDTH):
                percentage = MAXWIDTH/width
                frame = cv2.resize(frame, (0,0), fx=percentage, fy=percentage)
                imageResized = True
                height, width, channels = frame.shape
            
            if BBOX_ALPHA:
                MIN_BBOX_AREA = height*width*(BBOX_ALPHA**2)
            else:
                MIN_BBOX_AREA = 0

            time1 = time.clock() - start_time
            print("--- IMAGE RESIZING COMPLETE: %s seconds ---" % time1)

            ## *** PREPARE FOR DETECTION *** ##
            image_np = frame.copy() # So as not to override original image in memory
            #cv2.imshow('image',image_np)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            ## *** ACTUAL DETECTION *** ##
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            endTime = time.process_time()
            time2 = time.clock() - start_time
            print("--- DEEP LEARNING RESULTS ACHIEVED: %s seconds ---" % time2)
            #print(np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores))
            boxesDL = get_boxes_from_DL(np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, min_score_thresh=BBOX_CONFIDENCE)
            unpolar_boxes(boxesDL, image_np)
            
            ## *** DRAWING AND MIN AREA CHECKS *** ##
            tooSmallCount = 0
            for box in boxesDL:
                boxColour = (0, 255, 0)
                if ((box["xmax"]-box["xmin"])*(box["ymax"]-box["ymin"])) < MIN_BBOX_AREA: # Fails to pass min area check
                    boxColour = (0, 0, 255)
                    tooSmallCount += 1
                cv2.rectangle(image_np, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), boxColour, 3)
            time3 = time.clock() - start_time
            print("--- BOXES DRAWN: %s seconds ---" % time3)

            lenBoxesDL = len(boxesDL)
            print("Boxes found: ", lenBoxesDL)

            timeNoImportOrResizing = time3-time1 # For use later

            ## *** GROUND-TRUTH STATS *** ##
            if groundTruths:

                ### CONSTRUCT LOCAL IMAGE AGGREGATE STAT STUFFS ###
            
                iousTotal = []
                iousRecognized = []

                MRraw = 0

                ### RESIZE GROUND TRUTHS IF NECESSARY ###
                lenGTs = len(groundTruths)
                if imageResized:
                    i = 0
                    while i < lenGTs:
                        gtbox = groundTruths[i]
                        gtbox["xmin"] = int(gtbox["xmin"]*percentage)
                        gtbox["xmax"] = int(gtbox["xmax"]*percentage)
                        gtbox["ymin"] = int(gtbox["ymin"]*percentage)
                        gtbox["ymax"] = int(gtbox["ymax"]*percentage)
                        i += 1

                ### GET GROUND-TRUTHS BOXES ###
                for gtbox in groundTruths:
                    # Draw all the gt boxes, for fun
                    xmin, ymin, xmax, ymax = decodeBoxFromDict(gtbox)
                    cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (120, 120, 120), 1)

                    # Also, add a "detections" key-value pair to them.
                    gtbox["detections"] = 0

                ### PERFORM GROUND-TRUTH BOX OPERATIONS  ###
                for box in boxesDL:
                    # Find closest box to IoU box (Manhattan distance)
                    gtbox = closestGroundTruthBox(box, groundTruths)

                    # Calculate and insert IoU
                    IoUtemp = bb_intersection_over_union(box, gtbox)
                    iousTotal.append(IoUtemp)
                    if IoUtemp >= IOU_THRESH_MIN:
                        iousRecognized.append(IoUtemp)
                        # CRITICAL HIT!
                        gtbox["detections"] += 1
                    else: # Misidentification
                        MRraw += 1

                    # Draw!
                    xmin, ymin, xmax, ymax = decodeBoxFromDict(gtbox)
                    cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (255, 105, 180), 2)

            ## *** FINAL CALCULATIONS AND THE BIG REVEAL *** ##
            
                ### STATS DEPENDENT ON GROUND-TRUTH ###
                print('Actual Bounding Box Count (from Ground Truth): %d' % lenGTs)

                averageGT = np.average(iousTotal)
                print('Average IoU (Total): {}'.format(averageGT))
                print(iousTotal)

                averageGTRec = np.average(iousRecognized)
                print('Average IoU (Recognized @ >= {}): {}'.format(IOU_THRESH_MIN, averageGTRec))
                print(iousRecognized)

                countIousRecognized = 0
                countIousDuplicated = 0
                countIousNotDetected = 0
                for gtbox in groundTruths:
                    numDetected = gtbox["detections"]
                    if numDetected == 0: # Case 1: Undetected ("bad case")
                        countIousNotDetected += 1
                    elif numDetected == 1: # Case 2: One detection of GT ("good case")
                        countIousRecognized += 1
                    else: # Case 3: Multiple detections of same GT ("meh case")
                        dup = numDetected-1
                        countIousRecognized += 1
                        countIousDuplicated += dup

                #### Remove the too-smalls ####
                countIousRecognized -= tooSmallCount
                countIousNotDetected += tooSmallCount
                
                lenGTsFloat = float(lenGTs)
                RR = (countIousRecognized/lenGTsFloat)
                print('Recognition Rate: %f' % RR)

                DR = (countIousDuplicated/lenGTsFloat)
                print('Duplication Rate: %f' % DR)

                UR = (countIousNotDetected/lenGTsFloat)
                print('Undetected Rate: %f' % UR)

                lenIoUsTotal = len(iousTotal)
                MR = ((MRraw/float(lenIoUsTotal)) if lenIoUsTotal > 0 else np.inf)
                print('Misidentification Rate: %f' % MR)

                precision = (countIousRecognized / float(countIousRecognized + MRraw)) if (countIousRecognized + MRraw) > 0 else np.inf
                recall    = (countIousRecognized / float(countIousRecognized + countIousNotDetected)) if (countIousRecognized + countIousNotDetected) > 0 else np.inf
                try:
                    fscore = 2*(recall*precision)/(recall+precision) # DICE SIMILARITY COEFFICIENT!
                except Exception:
                    fscore = np.inf

                print('Precision: %f' % precision)
                print('Recall: %f' % recall)
                print('F1-Score (DSC): %f' % fscore)
                
                #### Add to global aggregate stats lists ####
                if np.isinf(precision):
                    globalPrecision.append(0)
                else:
                    globalPrecision.append(precision)
                    
                if np.isinf(recall):
                    globalRecall.append(0)
                else:
                    globalRecall.append(recall)
                    
                if np.isinf(fscore) or np.isnan(fscore):
                    globalFscore.append(0)
                else:
                    globalFscore.append(fscore)

                # Add to globals
                if not np.isnan(averageGTRec):
                    globalIOUsRecognized.extend(iousRecognized)
                if not np.isnan(averageGT):
                    globalIOUsTotal.extend(iousTotal)
                globalRR.append(RR)
                globalDR.append(DR)
                if not np.isinf(MR):
                    globalMR.append(MR)

            ### STATS UNRELATED TO GROUND-TRUTH ###
            
            # Timing results for the first always involve "loading"
            # So, don't let it contribute to time
            if hasProcessedFirst:           
                print("Time taken (post image import and applicable scaling): %s seconds" % timeNoImportOrResizing)
                globalTimeWithoutPictureImportAndResizingTime.append(timeNoImportOrResizing)

                globalTimeTotal.append(time3)
                
            hasProcessedFirst = True # At this point the first is processed

            ## *** SHOW ME THE BERRY *** ##
            if not args.quiet:
                cv2.imshow('object detection', image_np)
                keyPressed = cv2.waitKey(0)
                cv2.destroyAllWindows()

            ## *** EXPORT *** ##
            if not args.dry_run:
                imgSaveExt = filenameExt

                ### CREATE/ADJUST OUTPUT FOLDER IF NECESSARY ###
                output_folder = outfolder # Set the local save folder to the global save fodler
                # Want save in a new folder? Reset local save folder!
                if args.save_in_new_folder:
                    output_folder = os.path.join(output_folder, filenameBasename)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                ### SAVE OUTPUT IMAGE ###
                saveFilename = filenameBasename + "_marked" + imgSaveExt
                fullpath = os.path.join(output_folder,saveFilename)
                cv2.imwrite(fullpath, image_np)

                ### SAVE IMAGE AGGREGATE STATS ###
                if args.save_stats:
                    #### What was in STDOUT as text ####
                    with open(os.path.join(output_folder, filenameBasename + '.txt'), "w+") as outputstatsf:
                        outputstatsf.write('***** STATS FOR {} *****\n'.format(filename))
                        outputstatsf.write("--- IMAGE READ COMPLETE: %s seconds ---\n" % time0)
                        outputstatsf.write("--- IMAGE RESIZING COMPLETE: %s seconds ---\n" % time1)
                        outputstatsf.write("--- DEEP LEARNING RESULTS ACHIEVED: %s seconds ---\n" % time2)
                        outputstatsf.write("--- BOXES DRAWN: %s seconds ---\n" % time3)
                        outputstatsf.write("Boxes found: {}\n".format(lenBoxesDL))
                        if groundTruths:
                            outputstatsf.write('Actual Bounding Box Count (from Ground Truth): %d\n' % lenGTs)
                            outputstatsf.write('Average IoU (Total): {}\n'.format(averageGT))
                            outputstatsf.write('{}\n'.format(iousTotal))
                            outputstatsf.write('Average IoU (Recognized @ >= {}): {}\n'.format(IOU_THRESH_MIN, averageGTRec))
                            outputstatsf.write('{}\n'.format(iousTotal))
                            outputstatsf.write('Recognition Rate: %d\n' % RR)
                            outputstatsf.write('Duplication Rate: %d\n' % DR)
                            outputstatsf.write('Undetected Rate: %d\n' % UR)
                            outputstatsf.write('Misidentification Rate: %d\n' % MR)
                            outputstatsf.write("Time taken (post image import and applicable scaling) per bounding box recognized: {} seconds\n".format(timeNoImportOrResizingNormPerRec))
                            outputstatsf.write("Time taken per bounding box recognized: {} seconds\n".format(timeNormPerRec))
                        outputstatsf.write("Time taken (post image import and applicable scaling): %s seconds\n" % timeNoImportOrResizing)
                        outputstatsf.write("Time taken (post image import and applicable scaling) per detected box: %s seconds" % timeNoImportOrResizingNormPerBoxesDetected)

                    #### Found boxes as CSV ####
                    with open(os.path.join(output_folder, '{}_boxes.csv'.format(filenameBasename)), 'w+') as f:
                        f.write("class,probability,ymin,xmin,ymax,xmax,detections\n")
                        for box in boxesDL:
                            f.write("{},{},{},{},{},{},{}\n".format(box["class"], box["score"], box["ymin"], box["xmin"], box["ymax"], box["xmax"], box["detections"]))

                ### SAVE PER-IMAGE AGGREGATE STATS ###
                if args.save_aggregate_statistics:
                    with open(os.path.join(outfolder, globalStatsCSVFilename), 'a') as f:
                        if groundTruths:
                            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(filename,time0,time1,time2,time3,lenBoxesDL,lenGTs,averageGT,averageGTRec,RR,DR,UR,MR,precision,recall,fscore))
                        else:
                            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(filename,time0,time1,time2,time3,lenBoxesDL,None,None,None,None,None,None,None,None,None,None))

### PRINT AND SAVE GLOBAL AGGREGATE STATS ###
print("***** AGGREGATE STATISTICS FOR {} *****".format(infolder))

if args.save_aggregate_statistics_readout:
    runningString = "***** AGGREGATE STATISTICS FOR {} *****\n".format(infolder)

argsString="Input Folder: {}\nOutput Folder: {}\nGround-truth Folder: {}\nMin Area Percent (alpha): {}\nMin Confidence: {}\nMin Jaccard (IoU): {}\nMax Image Width: {}\n".format(infolder,outfolder,xmlfolder,BBOX_ALPHA,BBOX_CONFIDENCE,IOU_THRESH_MIN,MAXWIDTH)
print(argsString)

if args.save_aggregate_statistics_readout:
    runningString += argsString

if globalIOUsTotal:
    globalIOUs = np.array(globalIOUsTotal)
    print("IOU (Total) Global Mean:", np.mean(globalIOUs))
    print("IOU (Total) Global Std. Dev:", np.std(globalIOUs))
    if args.save_aggregate_statistics_readout:
        runningString += "IOU (Total) Global Mean: {}\n".format(np.mean(globalIOUs))
        runningString += "IOU (Total) Std. Dev: {}\n".format(np.std(globalIOUs))

if globalIOUsRecognized:
    globalIOUs = np.array(globalIOUsRecognized)
    print("IOU (Recognized) Global Mean:", np.mean(globalIOUs))
    print("IOU (Recognized) Global Std. Dev:", np.std(globalIOUs))
    if args.save_aggregate_statistics_readout:
        runningString += "IOU (Recognized) Global Mean: {}\n".format(np.mean(globalIOUs))
        runningString += "IOU (Recognized) Global Std. Dev: {}\n".format(np.std(globalIOUs))

if globalTimeTotal:
    globalTime = np.array(globalTimeTotal)
    print("Total Processing Time Global Mean:", np.mean(globalTime))
    print("Total Processing Time Global Std. Dev:", np.std(globalTime))
    if args.save_aggregate_statistics_readout:
        runningString += "Total Processing Time Global Mean: {}\n".format(np.mean(globalTime))
        runningString += "Total Processing Time Global Std. Dev: {}\n".format(np.std(globalTime))

if globalTimeWithoutPictureImportAndResizingTime:
    globalTime = np.array(globalTimeWithoutPictureImportAndResizingTime)
    print("Processing Time (no import and applicable resizing) Global Mean:", np.mean(globalTime))
    print("Processing Time (no import and applicable resizing) Global Std. Dev:", np.std(globalTime))
    if args.save_aggregate_statistics_readout:
        runningString += "Processing Time (no import and applicable resizing) Global Mean: {}\n".format(np.mean(globalTime))
        runningString += "Processing Time (no import and applicable resizing) Global Std. Dev: {}\n".format(np.std(globalTime))

if globalRR:
    rate = np.array(globalRR)
    print("Recognition Rate Global Mean:", np.mean(rate))
    print("Recognition Rate Global Std. Dev:", np.std(rate))
    if args.save_aggregate_statistics_readout:
        runningString += "Recognition Rate Global Mean: {}\n".format(np.mean(rate))
        runningString += "Recognition Rate Global Std. Dev: {}\n".format(np.std(rate))

if globalDR:
    rate = np.array(globalDR)
    print("Duplication Rate Global Mean:", np.mean(rate))
    print("Duplication Rate Global Std. Dev:", np.std(rate))
    if args.save_aggregate_statistics_readout:
        runningString += "Duplication Rate Global Mean: {}\n".format(np.mean(rate))
        runningString += "Duplication Rate Global Std. Dev: {}\n".format(np.std(rate))

if globalMR:
    rate = np.array(globalMR)
    print("Misidentification Rate Global Mean:", np.mean(rate))
    print("Misidentification Rate Global Std. Dev:", np.std(rate))
    if args.save_aggregate_statistics_readout:
        runningString += "Misidentification Rate Global Mean: {}\n".format(np.mean(rate))
        runningString += "Misidentification Rate Global Std. Dev: {}\n".format(np.std(rate))

if globalPrecision:
    prec = np.array(globalPrecision)
    print("Precision Global Mean:", np.mean(prec))
    print("Precision Global Std. Dev:", np.std(prec))
    if args.save_aggregate_statistics_readout:
        runningString += "Precision Global Mean: {}\n".format(np.mean(prec))
        runningString += "Precision Global Std. Dev: {}\n".format(np.std(prec))

if globalRecall:
    rec = np.array(globalRecall)
    print("Recall Global Mean:", np.mean(rec))
    print("Recall Global Std. Dev:", np.std(rec))
    if args.save_aggregate_statistics_readout:
        runningString += "Recall Global Mean: {}\n".format(np.mean(rec))
        runningString += "Recall Global Std. Dev: {}\n".format(np.std(rec))

if globalFscore:
    fs = np.array(globalFscore)
    print("F1-Score (DSC) Global Mean:", np.mean(fs))
    print("F1-Score (DSC) Global Std. Dev:", np.std(fs))
    if args.save_aggregate_statistics_readout:
        runningString += "F1-Score (DSC) Global Mean: {}\n".format(np.mean(fs))
        runningString += "F1-Score (DSC) Global Std. Dev: {}\n".format(np.std(fs))

if args.save_aggregate_statistics_readout and not args.dry_run:
    with open(os.path.join(outfolder, globalStatsTXTFilename), 'w+') as f:
        f.write(runningString)
