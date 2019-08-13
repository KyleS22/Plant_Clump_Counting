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
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr

def countdiff(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.average((y_true - y_pred))

def abscountdiff(y_true, y_pred):

    return abs(countdiff(y_true, y_pred))

def accuracy(y_true, y_pred):
    """
    Returns the accuracy of the model
    
    :param y_true: The true counts
    :param y_pred: The predicted counts
    :returns: The accuracy of the model
    """
    

    predicted_vals = np.rint(y_pred)
    correct = (predicted_vals == y_true)

    return (correct.sum() / correct.size) * 100

def pearson_r_square(y_true, y_pred):
  
    return pearsonr(y_true, y_pred)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Returns the mean absolute percentage error between y_true and y_pred.
    
    :param y_true: The true counts
    :param y_pred: The predicted counts
    :returns: The mean absolute percentage error between the true and predicted counts
    """
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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
 
def conf_matrix(y_true, y_pred):
    """
    Returns a confusion matrix for the model
    
    :param y_true: The true counts
    :param y_pred: The predicted counts
    :returns: An array representing a confusion matrix
    """
    

    return confusion_matrix(y_true, np.rint(y_pred))
