# Test Utils
This package contains modules that define functionality to aid in the testing of various models.  This includes computation of metrics, loading and saving model files and results, and other related behaviours.

# Modules

## `create_prediction_csv.py`
This module contains functions that will run a set of images through a given model and store the outputs in a set of csv files.

It contains one public function `create_prediction_csv()` that takes the following parameters:

- `real_counts_csv`: The path to the csv containing row numbers and ground truth counts as its columns
- `path_to_clump_images`: The path to the folder containing clump images to predict
- `path_to_model`: The path to the model save file 
- `param out_path`: The path to the directory to store the outputs in
- `param model_type`: The type of model that is being loaded.  Possible values are "CNN", "ENCODER", "CapsNet", "FFT", "LBPH", "GLCM".  Default is "CNN"
- `path_to_weights:` The path to the model weights, if it is a CNN type model.  Default is None.


Running this function will create a set of csv files in the `out_path` in a folder called `per_row_results`.  The csv files will be formatted as follows:

The file names will be the row number that the results are for: `11.csv` is the results for row 11

```
stitch_name, predicted_count, true_count
AAAAAAAAA,X,Y
AAAAAAAAA,X,Y
.... etc
```

## `evaluate_model.py`

This module contains functionality for evaluating model performance based on the per row results output by the `creat_prediction_csv.py` module.

There is one public function `evaluate_model()` that takes in the following parameters:
- `path_to_results_dir`: This is the path to the per row results produced by `create_prediction_csv.py`
- `out_path`: This is the path to the directory to store results in


This module creates a few files:
	- A `model_results.csv` file will be created containing one row with the values for various metrics on all the predictions
	- A `model_mean_results.csv` file will be created containing one row with the values for various metrics on the mean of the counts for each row
	- A set of updated per row results csv files, with an additional column containing the absolute error of each row

## `metrics.py`
This module contains definitaions for various metrics used to evaluate the models.

The available metrics are:
- countdiff
- accuracy
- pearson r
- mean absolute percentage error
- mean squared error
- mean absolute error
- r square
- confusion matrix

## `utils.py` 
This module contains various utility functions for loadin in different types of models and saving and managing test results


## `validation.py`
This module contains functionality for running a set of clump images through a model to get validation results.  This is intended for use with clump images for which each image there is a known count.  When evaluating a model on the outputs of an object detection algorithm, use `evaluate_model.py` instead. 
