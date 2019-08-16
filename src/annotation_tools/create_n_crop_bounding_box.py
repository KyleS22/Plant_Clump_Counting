import numpy as np
import matplotlib.pyplot as plt 
import skimage.io as io
import matplotlib.patches as pt
import os as os
import sys
from math import ceil

def show_bounding_boxes(dir_name="Umair"):
    
    #Parameter:dir_name assumes that both image and annotation directories are in your pwd
    #The function takes the direcotry name and creates bounding boxes around the annoted 
    #clumped plants.
    
    
    images_path = os.path.join('.', dir_name)
    
    if not os.path.exists(images_path):
        print("Directory Doesn't Exist\nExiting the Program")

    for root, dirs, files in os.walk(images_path):
        for filename in files:
            image = io.imread(root+'//'+filename)
            annotations_file = open(dir_name.replace(dir_name, dir_name+'_annotations')+'/'+filename.replace(".JPG",".txt"),"r")
            fig = plt.figure(figsize=(10,10))
            boxes = annotations_file.readlines()
            for box in boxes:
                coordinates = np.array(box.split())
                coordinates = coordinates.astype(np.float)
                x = (coordinates[1] - coordinates[3]/2)*image.shape[1]
                y = (coordinates[2] - coordinates[4]/2)*image.shape[0]
                rect = pt.Rectangle((x,y),coordinates[3]*image.shape[1],coordinates[4]*image.shape[0],linewidth=2,edgecolor='y',facecolor='none')
                ax2 = fig.add_subplot(111, aspect='equal')
                ax2.add_patch(rect)
            plt.imshow(image)

def cropped_bounding_boxes(dir_name='Umair',filename='IMG_4426.JPG'):
    
    #Parameter:dir_name is the name of the directory
    #Parameter:filename is the name of the image file
    #returns: cropped images array and the class_label array
    #The class_label_array will represent the TRUE class of same image on corresponding cropped_image array
    
    try:
        image = io.imread(dir_name+'/'+filename, plugin='matplotlib')
        annotations_file = open(dir_name + '_annotations' + '/' + filename.replace(".JPG", ".txt"), 'r')#dir_name.replace(dir_name.split('/')[0], dir_name.split('/')[0]+'_annotations')+'/'+filename.replace(".JPG",".txt"),"r")
    except:
        raise Exception("Cannot find the specified directory %s" & dir_name+'//'+filename)
        #sys.exit("Cannot Find the specified direcotry/Image %s" % dir_name + '//' + filename)
    
    boxes = annotations_file.readlines()
    cropped_images=[]
    image_labels = []
        

    for box in boxes:
        coordinates = np.array(box.split())
        coordinates = coordinates.astype(np.float)
        x = (coordinates[1] - coordinates[3]/2)*image.shape[1]

        y = (coordinates[2] - coordinates[4]/2)*image.shape[0]
        

        start_y = int(ceil(y))
        end_y = int(ceil(y+coordinates[4] * image.shape[0]))

        start_x = int(ceil(x))
        end_x = int(ceil(x+coordinates[3]*image.shape[1])) 
        

        cropped_images.append(image[start_y:end_y, start_x:end_x])
        

        image_labels.append(int(coordinates[0]+1))
        

    return cropped_images,image_labels, filename


# Function Call Test
#cropped_images,image_labels = cropped_bounding_boxes('Data/Training','IMG_4456.JPG')
#for i in range(0,len(cropped_images)):
#    fig = plt.figure()
#    plt.title("Clump Image: "+str(i+1)+" with: "+str(image_labels[i])+" Plants")
#    plt.imshow(cropped_images[i])
