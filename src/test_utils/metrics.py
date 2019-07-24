"""
File Name: metrics.py

Authors: {% <AUTHOR> %}

Date: 23-07-2019

Description: A module containing the metrics we will use to evaluate counting models

"""
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def accuracy(y_true, y_pred):
    """
    Returns the accuracy of the model
    
    :param y_true: The true counts
    :param y_pred: The predicted counts
    :returns: The accuracy of the model
    """
    

    predicted_vals = np.rint(y_pred)

    correct = (predicted_vals == y_true)

    return correct.sum() / correct.size

def mean_squared_error(y_true, y_pred):
    """
    Returns the MSE of the model
    
    :param y_true: The true counts
    :param y_pred: The predicted counts
    :returns: The MSE of the model
    """

    return mse(y_true, y_pred)#(np.square(np.array(y_true) - np.array(y_pred))).mean(axis=-1)

def mean_absolute_error(y_true, y_pred):
    """
    Returns the Mean Absolute Error of the model
    
    :param y_true: The true counts
    :param y_pred: The predicted counts
    :returns: The Mean Absolute Error of the model
    """

    return mae(y_true, y_pred)#np.sum(np.absolute(np.array(y_true) - np.array(y_pred)))
    

def r_square(y_true, y_pred):
    """
    Return the r_square metric for the model
    
    :param y_true: The true counts
    :param y_pred: The predicted counts
    :returns: The R squared value for the model
    """
    return r2_score(y_true, y_pred)
 
