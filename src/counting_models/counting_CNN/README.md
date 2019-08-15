# Encoder

This directory contains the definitions of the "CNN Counting model".  This is a model that id trained as a regression model for counting,

The `model.py` file contains class definitions for the model, namely the `CountingModel` class that represents the model.  It has the necessary funtions to train the model and use the model to predict various inputs.

The `train_runner.py` file is a command line program to facilitate starting the training process.  It takes the following parameters:
- `training_data_dir`: The directory containing training images.  It must be organized as detailed below
- `model_save_dir`: The name of the directory to save the model outputs in (saved weights and such). The default is 'model'.  This directory will be created if it does not exist, and if it does exist, a waringin message will be displayed.
- `validation_data_dir`: The directory containing validation images.  It must be organized as detailed below
- `batch_size`: The size of batches to use for training
- `num_epochs`: The number of epochs to train for
- `model_name`: The name of the model, used for naming output files

## Training and Validation data format

The training and validation images should be organized in the following directory structure.

```
training_dir/
	1/
		IMGx.png
		IMGx.png
		...
	2/
		IMGx.png
		IMGx.png
	...

	15/
		IMGx.png
		IMGx.png

validation_dir/
	1/
		IMGx.png
		IMGx.png
		...
	2/
		IMGx.png
		IMGx.png
	...

	15/
		IMGx.png
		IMGx.png

```

The subfolders of the main directory should be named as the ground truth count for the images inside of it.  So `training_dir/2/*` contains images that have two plants in a clump.

 
