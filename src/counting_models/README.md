# Counting Models

This package contains sub-packages with definitions of various counting models.

There are also a few command line scripts for convenience when using the counting models

# Counting CNN
This package defines modules related to defining and training a CNN for counting

# Encoder
This package defines modules related to defining and training a Convolutional Autoencoder for extracting image features as a pre-training step for counting.

# Traditional ML
This package defines modules related to defining and training Traditional Machine learning models (such as K-Nearest Neighbours classifiers) to count plants.


# Visualize Conf Matrix
This is a simple script that takes in a CSV representing a confusion matrix and turns it into a coloured plot for visualization purposes.
It takes the following parameters:
- `file_path`: The path to the confusion matrix CSV file.
- `--save`: If you want to save the generated plot, use this argument and specify a path including the name of the file.
- `--silent`:  By default, the program will display the matplotlib plot for you.  Use `--silent` if you want to skip the show.

# Validation Runner
The validation runner is a command line program that validates a given model against a validation set.

It has the following paramaters:
- `model_path`: The path to the saved model to load for validation
- `validation_data_dir`: The path to the directory containing the images to use as a validation set
- `test_result_path`: The path to the directory to save all result files
- `test_name`: The name for this test for file naming purposes
- `model_type`: The type of model being used.  Model types can be  "CNN", "ENCODER", "FFT", "GLCM", "LBPH":
     
- `--path_to_weights`:  If your model has a weights file, use this to provide the path to it. 


# Prediction Runner
This is a command line program that gets the predictions for a set of row images based on the output images of one of the object detection methods.

It has the following parameters:
- `model_path`: The path to the saved model to load for validation
- `validation_data_dir`: The path to the directory containing the images to use as a validation set
- `test_result_path`: The path to the directory to save all result files
- `model_type`: The type of model being used.  Model types can be  "CNN", "ENCODER", "FFT", "GLCM", "LBPH":
- `real_counts_path`: The path to the csv containing one column of row numbers, and one column of their ground truth counts.      
- `--path_to_weights`:  If your model has a weights file, use this to provide the path to it. 



