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
Because we have so few images (I know the dataset isn't actually on github, but trust me we don't have enough data) we decided to take a stab at synthetic images.  Those images are not sorted when they are generated, so we've got an entire script devoted to structuring synthetic images (or really any images with the naming scheme `COUNTNUMBER_IMAGENAME.png`) so that they work with our models. This one is pretty simple, it takes in a directory containing the images, and a directory to store the sorted images in.

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

## And Now the Feature Presentation
Finally some machine learning! Hold on to your hats, this one is exciting.  

There are a few model architectures presented here: CountingCNN, EncoderCNN,

### Counting CNN
This is a basic CNN set up to learn regression.  All of the model code is contained within `counting_CNN/`.  The classes used to represent the model is in `model.py`.  This directory also contains `train_runner.py`, which is a command line program that makes it easy to train the model.  It takes in a few parameters:
- ``--model_save_dir`` - This is the directory to store any of the model related output files in, including the trained model weights
- ``--validation_data_dir`` - This is the directory containing the images that should be used during the validation step of training
- ``--batch_size`` - This is the size of the batches to use during training
- ``--num_epochs`` - This is the number of epochs to train for
- `` --model_name`` - This is the name for this model.  It is used for naming output files so make it memorable.
- ``training_data_dir`` This is the directory to get the images to train the model on from.

All of the parameters above with ``--`` in front of them are optional.  They have default values and the program will run fine if you do not specify them, but it is a good idea to use them anyways, even just to be more explicit about how you are training your model.

Here is an example of training a model called ``my_awesome_model`` using a folder  `training_images` as the training data, and a folder `validation_images` as validation data.

```
python train_runner.py ----model_save_dir ./my_awesome_model --validation_data_dir validation_images --batch_size 32 --num_epochs 500 --model_name my_awesome_model training_images
```

It is a bit of a mouthful, but it gets the job done.  Something to note is that the model uses an early stopping technique, which means that even though I specify to train for 500 epochs, it will stop training early if the model decides that there has not been enough improvement in a while.  Once this is done, the trained model will be saved in the directory you specified, and it will be all trained up and ready to go!

### Encoder
This is a model that first trains an autoencoder on the images, and then trains a regression network using the weights from the autoencoder.  To run it, follow the instructions for the Counting CNN, except run the `train_runner.py` script in the `encoder/` directory instead.

# Top Level Scripts
There are a few convenience scripts in the `src/` directory for training and evaluating the models.  The `compare_models.py` script can be used to print out a table comparing the test results of different models.  It takes in a list of paths to get results csv files from, or you can specify the ``--is_dir`` option and pass in a single directory containing many csv files with results for different models.

So if I have a directory of results files:
```
 Results/
   model1.csv
   model2.csv
   ...
```
I can compare them all like this:
```
python compare_models.py --is_dir Results
```

To actually generate the results, you need to run the `validation_runner.py` script.  This takes in a few parameters:
- ``--real_counts_path`` - This is only used when using the ``--sys_test`` option
- ``--path_to_weights`` - This is the path to the trained model weights to use with the model
- ``model_path`` - This is the path to the trained model directory, where all of the trained model outputs are stores.  
- ``valdiation_data_dir`` - This is the path to the images to use for validation
- ``test_result_path`` - This is the place to store the test result csv file
- ``test_name`` - The name of the test (used for naming outputs)
- ``model_type`` - This is used to distingush the different models.  So if we are validting the CountingCNN we would use CNN here

As an example, say I want to validate the Encoder model.  Here is how I would do that
```
python validation_runner.py model_save_path/my_model.json validation_images results_dir test_name ENCODER --path_to_weights model_save_path 
```

The ``--system_test`` option is used to run a test on the whole counting system, but that does not work at the time of writing this.

## Shell Script for Easy Testing
Finally, there is a script called ``train_and_validate_model.sh``.  This is a convenience script that calls all of the other scripts required to train and validate a model.  

## Test Utils
There is a modlule called ``test_utils``.  This contains scripts that simply provide functionality for testing various modules.  It is designed to take any model as input, and should be flexible enough to use to compare many different types of models.





# Contribution Guide

To add functionality to the project, please create a new branch off of development for your new feature.  Completed features can be merged into development.  To merge with master, an approved review is required.


