from scipy.fftpack import fft
import skimage.io as io
import os as os 
import numpy as np
import skimage.filters as filt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle
from skimage.transform import rescale
from skimage.util import pad

import warnings 


class FourierTransformModel:
    """
    A model that uses features extracted from the Fourier spectrum of the image
    """

    def __init__(self, model_type=None, save_path=None):
        """
        Initialize the model
        
        :param model_type:      The type of model to use.  Could be one of the
                                following strings: 'KNN', 'SVC', 'GNB'
        :param save_path=None:  The path to the directory to save the model in
                                after training.
        :returns: None.         The model will be initialized with the correct
                                parameters
        """
        
        self.region_features=[]
        self.classes=[]
        self.test_region = []
        self.train_region = []
        self.train_classes_region = []
        self.test_classes_region =[]

        self.rotation_angles=[0,90,180,270]

        self.minimum=0.0
        self.maximum=0.0
        
        
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


    def _load_data_from_dir(self, path_to_data, train=True):
        """
        Load the images from the 'path_to_data' directory.
        
        :param path_to_data:    The path to the directory containing the images
        :param train:           Whether to load the data for training or for validation
                                purposes.  Default is True
        :returns:               None.  The images will be loaded and processed
                                for training.
        """
        
        for root, dirs, files in os.walk(path_to_data):
           
            for filename in files:
                cropped_image =  io.imread(os.path.join(root, filename))
                cropped_image = filt.unsharp_mask(cropped_image, radius=1, amount=4)

                image_class = root.replace("\\","/").split('/')[-1]
                if train:      
                    for angle in self.rotation_angles:
                        #print("Image: "+str(filename)+" Class: "+str(image_class)+" Angle: "+str(angle))
                        fourier = fft(cropped_image).flatten()
                        fourier = np.true_divide(fourier, np.count_nonzero(fourier))
                        hist,bins = np.histogram(fourier,bins=5)
                        
                        self.train_region.append(hist)
                        self.train_classes_region.append(int(image_class))
                else:
                    fourier = fft(cropped_image).flatten()
                    fourier = np.true_divide(fourier, np.count_nonzero(fourier))
                    hist,bins = np.histogram(fourier,bins=5)    

                    self.test_region.append(hist)
                    self.test_classes_region.append(int(image_class))

    def fit(self, train_data_dir):
        """
        Fit the model to the data in the given directory.
        
        :param train_data_dir:  The directory containing the images to train on
        :returns:               None.  The trained model will be saved to the
                                `save_path` that was initialized.
        """
        self._load_data_from_dir(train_data_dir, train=True)
        self.model.fit(self.train_region, self.train_classes_region)
        pickle.dump(model, open(os.path.join(self.save_path, self.model_type.upper(), "FFT_model.sav"), 'wb'))
 
    def load_model(self, path_to_model):
        """
        Load the model from a save file at the given path
        
        :param path_to_model: The path to the model save file
        :returns: The model will be initialized to be used for testing
        """
        self.model = pickle.load(open(path_to_model, 'rb'))
    
    def predict(self, image):
        """
        Predict the number of plants in the given image
        
        :param image:   The image to predict from.  Note that the image must be
                        loaded correctly using the 'prepare_input_from_file()' function.
        :returns: The predicted number of plants in the image
        """
        
        return self.model.predict(image)

    def predict_generator(self, data_dir, num=0):
        """
        Get predictions from the model on all images in the given directory.
        
        :param data_dir:    The directory to get the images from
        :returns:           The predicted number of plants in all of the images in the
                            given directory.
        """
        self._load_data_from_dir(data_dir, train=False)

        return self.predict(self.test_region)

    def validate(self, data_dir):
        """
        Get the predictions and true classes for the given folder of images
        
        :param data_dir: The directory to get the images from
        :returns: A tuple containing (y_true, y_pred)
        """

        self._load_data_from_dir(data_dir, train=False)
        
        return (self.predict(self.test_region), self.test_classes_region)

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
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fourier = fft(padded).flatten()
            fourier = np.true_divide(fourier, np.count_nonzero(fourier))
    
            hist,bins = np.histogram(fourier,bins=5)    

 
        return [hist]

