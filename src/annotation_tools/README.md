# Annotation Tools
This package contains modules that define functionality for dealing with
annotated data.

# Create Cropped Image Dataset
This is a command line program that will crop a set of images according to the
bounding boxes provided in an annotation file.  The annotations must be in YOLO
format.

The program has the following paramaters:

- `-i`: A list of directory names that will contain the images to process.  Assumes that the annotations for the images are in a folder with the same name followed by `_annotations`.  Ex) If the images are in `Images`, then the annotations are in `Images_annotations`.  These directories are listed on the command line as like so: `python create_cropped_image_dataset.py -i dir1 -i dir2 -i dir3 out_dir`

- `output_dir`: The directory to store the cropped images


# Split Images For Annotators
A command line program to divide a set of images into different groups for
annotation.

It has the following parameters:
- `input_dir`:  The directory to get the image from
- `output_dir`: The directory to store the sorted images in
- `num_raters`: The number of raters/annotators to divide the images between
- `overlap`:    The amount of overlap in images between raters.  Use this for
                inter-rater agreement studies.
- `-n`:         A list of rater names, for naming the sorted output
                directories.  Use as follows `python
                split_images_for_annotators.py input_dir outout_dir 5 10 -n
                Alice -n Bob -n Charlie


