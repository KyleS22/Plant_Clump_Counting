import matplotlib.pyplot as plt
import skimage.util as util
import skimage.io as io
import numpy as np
import os as os 
import skimage as sk
import skimage.color as cl
import skimage.filters as filt 
import scipy.ndimage.filters as fil
import skimage.morphology as morph
import skimage.segmentation as seg
import skimage.measure as measure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import skimage.feature as ft
import skimage.exposure as exp
from sklearn.model_selection import train_test_split
from skimage.transform import resize,rotate
import pickle
from skimage.transform import rescale
from skimage.util import pad

import warnings


class LBPHModel:
    """
    A machine learning model for coutning plants based on LBPH features.
    """

    def __init__(self, model_type=None, save_path=None):
        """
        Initialize the model
        
        :param model_type: The type of model to use.  Must be one of 'KNN','SVC', 'GNB'
        :param save_path=None: The path to save the trained model in.
        :returns: None. The model will be initialized.
        """
        
        self.LBPH_features=[]       
        self.classes = []
        self.test_LBPH = []
        self.train_LBPH = []
        self.train_classes_LBPH = []
        self.test_classes_LBPH = []

        self.rotation_angles = [0, 90, 180, 270]

        self.save_path = save_path
        if model_type is None:
            self.model = None
        elif model_type.upper() == "KNN":
            self.model = KNeighborsClassifier(n_neighbours=5)
        elif model_type.upper() == "SVC":
            self.model = svm.SVC(kernel='rbf', C=50)
        elif model_type.upper() == "GNB":
            self.model = GaussianNB()
        else:
            raise Exception("Model type not supported")

    def get_LBP(self, training_image):
        """
        Get the LBP features of the given image
        
        :param training_image: The image to get LBP features
        :returns: The lbp features of the image
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            training_image = cl.rgb2gray(training_image)
            Linv_LBP_image=0
            Var_LBP_image=0
            Linv_iterative_histogram=[]
            Var_iterative_histogram=[]
            iterative_histogram=[]
        
            Linv_LBP_image = ft.local_binary_pattern(training_image,P=32,R=3,method='uniform')
            Var_LBP_image = ft.local_binary_pattern(training_image,P=32,R=3,method='var')    
            Linv_iterative_histogram,notneeded = np.histogram(Linv_LBP_image,bins=10,range=(0,9))
            Var_iterative_histogram,notneeded = np.histogram(Var_LBP_image,bins=16,range=(0,7000))
            iterative_histogram = np.concatenate((Linv_iterative_histogram,Var_iterative_histogram),axis=None)
            
        return iterative_histogram



    def _load_data_from_dir(self, path_to_data, train=True):
        """
        Load the images in the given directory for training or testing.
        
        :param path_to_data:    The path to the directory containing the data
        :param train=True:      Whether to load the data as training or validation
                                data.  Default is true.
        :returns: None.  The data will be loaded for use with the model.
        """
        for root, dirs, files in os.walk(path_to_data):
            for filename in files:
                cropped_image = sk.img_as_float(io.imread(root+'/'+filename))
                image_class = root.replace("\\","/").split('/')[-1]
                if train:
                    for angle in self.rotation_angles:
                        rotated_image = exp.rescale_intensity(rotate(cropped_image,angle))
                       # print("Training Image: "+str(filename)+" Class: "+str(image_class)+" Angel:"+str(angle))
                        self.train_LBPH.append(self.get_LBP(rotated_image))
                        self.train_classes_LBPH.append(int(image_class))
                else:
                    self.test_LBPH.append(self.get_LBP(cropped_image))
                    self.test_classes_LBPH.append(int(image_class))

       
    def fit(self, train_data_dir):
        """
        Fit the model to the training data
        
        :param train_data_dir:  The directory to get the training data from
        :returns: None.         The model will be trained and saved in the chosen
                                directory from initialization.
        """
        self._load_data_from_dir(train_data_dir, train=True)
        self.model.fit(self.train_GLCM, self.train_classes_GLCM)
        pickle.dump(model, open(os.path.join(self.save_path, self.model_type.upper(), "LBPH_model.sav"), 'wb'))
 
    def load_model(self, path_to_model):
        """
        Load a pre-trained model from the given path/
        
        :param path_to_model: The path to the saved model file
        :returns: None.  The model will be loaded for use.
        """
        self.model = pickle.load(open(path_to_model, 'rb'))

    def predict(self, image):
        """
        Predict the count for the given image.
        
        :param image: The image to predict for
        :returns: The predicted number of plants in the image
        """
        
        return self.model.predict([image])

    def predict_generator(self, data_dir, num=0):
        """
        Predict the number of plants in each image in the given directory.
        
        :param data_dir: The directory to get the data from.
        :returns: The predictions of all the images in the given directory.
        """
        self._load_data_from_dir(data_dir, train=False)
        return self.model.predict(self.test_LBPH)

    def prepare_input_from_file(self, file_path, target_image_size=(112, 112)):
        """
        Loads and processes the given file for input to the model
        
        :param file_path: The path to the file
        :returns: The loaded and processed image, ready to use in the model
        """
        # load the image
        
        img = io.imread(file_path)
        
        
        if max(img.shape) > target_image_size[0]:
            # Get scaling factor 
            scaling_factor = target_image_size[0] / max(img.shape)

            # Rescale by scaling factor
            img = rescale(img, scaling_factor, multichannel=True)
        
                # pad shorter dimension to be 112
        pad_width_vertical = target_image_size[0] - img.shape[0]
        pad_width_horizontal = target_image_size[0] - img.shape[1]
        
        
        pad_top = int(np.floor(pad_width_vertical/2))
        pad_bottom = int(np.ceil(pad_width_vertical/2))
        pad_left =  int(np.floor(pad_width_horizontal/2))
        pad_right = int(np.ceil(pad_width_horizontal/2))

        padded = pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
        
       
         
        return self.get_LBP(padded)




