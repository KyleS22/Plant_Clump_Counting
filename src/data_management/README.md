# Data Management
This package contains modules that facilitate the easy sorting and general management of the image datasets.

# Create Train and Test Dataset
`create_train_and_test_dataset.py` contains functionality for producing a training and testing dataset from a given set
of images.  Assumes that the images are structured as outlined in the [Data Organization](#Data-Organization) section below.  It can be run as a command line program with the following parameters:

- `data_dir`:  This is the directory to get the data from
- `train_dir`: This is the path to the new directory to store training images in.  The directory must not exist
               beforehand unless you use the `--overwrite` option, it will be created for you
- `test_dir`:  This is the path to the new directory to store testing images in.  The directory must not exist
               beforehand unless you use the `--overwrite` option, it will be created for you

- `--overwrite`:  This is an optional flag that detemines whether to overwrite the training and testing data in the
                  given `train_dir` and `test_dir`.  Use this if you want to re-use the old training and testing
                  directories and DO NOT WANT TO KEEP THEIR DATA.  THIS WILL OVERWRITE ALL IMAGES IN THE TRAIN_DIR AND
                  TEST_DIR.

- `--test_size`:  The size of the testing set, as a floating point number between 0 and 1.  Represents the percent of
                  images to use for the test set.  Default is 0.2


# Downgrade iPhone Images
`downgrade_iphone_images.py` contains functionality for matching the histogram and resolution of the iPhone dataset to
the GoPro dataset. Assumes that the images are structured as outlined in the [Data Organization](#Data-Organization) section below. It can be run as a a command line program with the following parameters:

- `source_dir`:     The directory to get the iPhone images from
- `target_dir`:     The directory containing the target images to match to (the GoPro images)
- `output_dir`:     The directory to store the downgraded images in

# Resize Images
`resize_images.py` contains functionality for resizing a set of images to a predefined size, using padding to keep the
aspect ratio.  Assumes that the images are structured as outlined in the [Data Organization](#Data-Organization) section below.

It can be run as a command lin eprogram with the following parameters:

- `input_dir`: The directory to get the images from to be resized.  

Thats it!  The resized images will be stored in the same directory as the `input_dir` is located, in a folder with the
same name as the `input_dir`, with `_resized` appended to it.  So `python resize_images.py ~/data/images_dir` will
results in a new folder `~/data/images_dir_resized` that contains the resized images.


# Sort Synthetic Data
`sort_synthetic_data.py` contains functionality for sorting synthetic images into the structure outline in the [Data
Organization](#Data-Organization) section.  It assumes that the images to be sorted are in one folder with the following
structure:

```
synthetic_images/
    XX_YYYYYY.png
    XX_YYYYYY.png
    ...

```
Where `XX` represents the label of the image (the number of plants in the image), and `YYYYYY` represents the image
filename.

This can also be run as a command line program with the following parameters:

- `data_dir`: The directory to get the synthetic images from
- `out_dir`: The directory to store the sorted images in

# Data Organization
Images should be organized into subdirectories named after the labels of the images contained within:

```
images/
    1/
        IMG1.png
        IMG2.png
        ...
    2/
        IMG1.png
        IMG2.png
    ...
```

In the above example, the `1/` folder contains images with clumps of one plant, the `2/` folder contains images with
clumps of two plants, etc.
