# Plant_Clump_Counting
Automated System for counting clumps of plants as part of a course project at the University of Saskatchewan.

## Contributors

Kyle Seidenthal - Masters student in image processing and computer vision at the University of Saskatchewan

Blanche Leyeza - 

Syed Umair Aziz - 

Riel Castro-Zunti - 

Mahdiyar Molahasani -




**Special Thanks**: Theron Cory for helping with data acquisition.

# Usage

## Environment
This project is developed using anaconda python.  Use the provided environment.yml to create the environment and activate it.  Detailed instructions are provided below.

You can see how to install anaconda python here: https://docs.anaconda.com/anaconda/install/

### Linux/Mac 
Make sure you're  in the main project folder and that you can see the environment.yml file.

```
conda env create -f environment.yml
conda activate plant_counting
```

Tada!  You are now in the environment that can be used to run all of the code here!

## Annotation Tools
The source directory contains a set of command line python programs for producing annotated image datasets.  There are two scripts here:

### Split Images for Annotators
 This is found in `split_images_for_annotators.py`.  This program takes in a list of rater names, an input directory of images, am output_directory for the split images, the total number of raters, and a number of images to overlap between raters.
 
 For example, say I had a set of images stored in `my_images/` and I wanted to split them between three raters: Alice, Bob and Charlie.  I also want to make sure that 15 of the images in the folder are annotated by each of the three raters, and the rest of the images are only annotated by one annotator.  I would use the program as follows:
 
 ```
 python split_images_for_annotators.py -n Alice -n Bob -n Charlie ./my_images ./output_dir 3 15
 ```

### Create Cropped Image Dataset
 This is found in `create_cropped_image_dataset.py`.  This program takes in a list of directory names and outputs a directory full of cropped images.  The images and bounding box annotation files should be structured as follows:
 
 ```
 data_dir/
    first_images/
        first_image1.png
        first_image2.png
        ...
     first_images_annotations/
        first_image1.txt
        first_image2.txt
     
     second_images/
        second_image1.png
        second_image2.png
        ...
     second_images_annotations/
        second_image1.txt
        second_image2.txt
      
 ```

The program can then be used as follows to create a set of cropped images in the `output_dir`:
```
python create_cropped_image_dataset.py -i data_dir/first_images -i data_dir/second_images output_dir
```




# Contribution Guide

To add functionality to the project, please create a new branch off of development for your new feature.  Completed features can be merged into development.  To merge with master, an approved review is required.


