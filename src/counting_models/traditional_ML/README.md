# Traditional Machine Learning
This package contains modules that define classes for traditional machine
learning approaches to counting the number of plants in an image.

Each module defines a set of models that use a different type of image feature.
Each model is trainable as one of the following types: 'KNN' (KNearestNeighbours), 'SVC' (Support Vector), 'GNB' (Gaussian Naive Bayes).

# GLCM Based
This module defines the `GLCMModel` class.  This defines a model that uses Gray
Level Co-occurance Matrices as the image feature to train on.  

# LBPH Based
This module defines the `LBPHModel` class.  This defines a model that uses
Local Binary Pattern Histograms as the image feature to train on.

# Fourier Based
This model defines the `FourierTransformModel` class, which defines a model
that uses the Fourier spectrum of the image to extract features to train on.

# Train Runner
The train runner module defines a command line program to easily train any of
the above models.  It has the following command line arguments:

- `training_data_dir`:      The directory to get training data from.
- `model_type`:             The type of model to use.  Must be one of 'KNN', 'SVC', 'GNB'
- `feature_type`:           The type of image feature to train on.  Must be one
                            'FFT', 'LBPH', 'GLCM'.
- `--model_save_dir`:       The directory to save the trained model file in.
- `--model_name`:           The name of the model for file naming purposes.
- `--validataion_data_dir`: The directory to get validation data from

# Training and Validation data format

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

 
