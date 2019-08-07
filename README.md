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

## Data Management Tools

Next up are a set of tools for dealing with the images, so that we can get them into an organized and uniform structure.  These can be found in `src/data_management`.

### Creating a Training and Testing Dataset

Because our images are organized in a specific way, and this is a collaborative project, it is useful for us to be able to generate a training and testing dataset from our images before we start training our models.  Thats where the `create_train_and_test_dataset.py` script is handy.

The script takes in an input directory of images, and the paths to the training and testing directories.  There is also a `--overwrite` option, which allows you to generate the training and testing sets into an existing directory.  **BE WARNED, IT WILL DELETE EXISTING DATA!**

The images must be in the following structure:
```
input_dir/
  1/
   IMG###.png
   ...
  2/
   IMG###.png
  3/
  ...
```

Where the numbered folders represent the class that the images within belong to.  So all the images with 2 plants are in `input_dir/2/`.

Here's an example of creating a train and test set from the images in `input_dir`:
```
python create_train_and_test_dataset.py input_dir output_training_dir output_testing_dir
```

Cool!

### Downgrade iPhone Images

Our dataset had two different types of images: low resolution zoomed out images from a GoPro camera, and high resolution zoomed in images from an iPhone.  These two image types are drastically different, and so we need to normalize them somehow.  It would be really cool if we could bring the GoPro images up to the quality of the iPhone images, but unfortunately we need to downsample the iPhone images to the resolution of the GoPro images.  This is done with the `downsgrade_iphone_images.py` script.

This script takes in a source directory containing iPhone images in the structure stated in the above section.  It also takes in a target directory that that contains the low resolution images we want the iPhone images to look like, and finally an output directory. 

Here's an example to convert the images in `source_dir` so that they match the images in `target_dir`.  The downsampled images will be stored in `out_dir`:
```
python downgrade_iphone_images.py source_dir target_dir out_dir
```

Boom! Now your shiny iPhone images look really bad!  Trust me this is a good thing.


### Sort Synthetic Data
Because we have so few images (I know the dataset isn't actually on github, but trust me we don't have enough data) we decided to take a stab at synthetic images.  Those images are not sorted when they are generated, so we've got an entire script devoted to structuring synthetic images (or really any images with the naming scheme COUNTNUMBER_IMAGENAME.png) so that they work with our models. This one is pretty simple, it takes in a directory containing the images, and a directory to store the sorted images in.

Here's an example:
```
python sort_synthetic_data.py unsorted_synthetic out_dir
```

And the images are now sorted

### Resize Images

Our images have a wide range of sizes (from about (25, 25) to (250, 50)).  This does not play well with our models, so we have to make sure the images are all the same size when they are input to the model.  We can't just resize them willy nilly though, otherwise we would lose the aspect ratio of the images and end up losing information.  Instead we need to rescale only images that are larger than our predetermined size, and make sure to scale all dimensions proportionally.  We then zero-pad any dimensions that are smaller than our predetermined size.  The script `resize_images.py` does just that.  Note that the default image size is (112, 112)

Here's an example of resizing all images in the `input_dir`:

```
python resize_images.py input_dir
```

Very anti-climactic, I know.  But now there should be a folder named `input_dir_rescaled` that contains all of your images just like `input_dir`, except these ones are all the same size!


# Contribution Guide

To add functionality to the project, please create a new branch off of development for your new feature.  Completed features can be merged into development.  To merge with master, an approved review is required.


