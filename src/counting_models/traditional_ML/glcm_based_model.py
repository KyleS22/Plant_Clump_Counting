
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

class GLCMModel:
    """
    A machine learning model for counting plants based on GLCM features.
    """

    def __init__(self, model_type=None, save_path=None):
        """
        Initialize the model
        
        :param model_type=None: The type of model to use.  Must be one of 'KNN', 'SVC', 'GNB'
        :param save_path=None: The path to the directory to save the model in
        :returns: None.  The model will be initialized for use.
        """
        
        self.GLCM_features=[]       
        self.classes = []
        self.test_GLCM = []
        self.train_GLCM = []
        self.train_classes_GLCM = []
        self.test_classes_GLCM = []

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

    def GLCM_feature(self, Image):
        """
        Get the GLCM features for the given image
        
        :param Image: The image to get GLCM features for
        :returns: The GLCM features for the given image.
        """
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Image =sk.img_as_ubyte(cl.rgb2gray(Image))
        
        glcm_feature=0
        glcms = ft.greycomatrix(Image,distances=[2,4,8,16,32,48,64,128,256,512],angles=[0.785398,1.5708,3.142],normed=True)
        #glcm_feature_1 = ft.greycoprops(glcms, prop='contrast')
        glcm_feature_2 = ft.greycoprops(glcms, prop='dissimilarity')
        glcm_feature_3 = ft.greycoprops(glcms,prop='correlation')
        
        #glcm_feature_1 = np.mean(glcm_feature_1,axis=1)
        glcm_feature_2 = np.mean(glcm_feature_2,axis=1)
        glcm_feature_3 = np.mean(glcm_feature_3,axis=1)
        
        
        glcm_feature = np.concatenate((glcm_feature_2,glcm_feature_3),axis=None) 
        #print(glcm_feature)
        return glcm_feature



    def _load_data_from_dir(self, path_to_data, train=True):
        """
        Load the data in the given directory for use with the model for
        training and validation.
        
        :param path_to_data:    The path to the directory containing the images to
                                load
        :param train:  Whether this data is to be loaded as training data
                            or validataion data.  Default is True
        :returns: None.  The data will be loaded for use with the model
        """
        for root, dirs, files in os.walk(path_to_data):
            for filename in files:
                cropped_image = sk.img_as_float(io.imread(root+'/'+filename))
                image_class = root.replace("\\","/").split('/')[-1]
                
                if train:
                    for angle in self.rotation_angles:
                        rotated_image = exp.rescale_intensity(rotate(cropped_image,angle))
                       # print("Training Image: "+str(filename)+" Class: "+str(image_class)+" Angel:"+str(angle))
                        self.train_GLCM.append(self.GLCM_feature(rotated_image))
                        self.train_classes_GLCM.append(int(image_class))
                else:
                    self.test_GLCM.append(self.GLCM_feature(cropped_image))
                    self.test_classes_GLCM.append(int(image_class))

       
    def fit(self, train_data_dir):
        """
        Fit the model to the training data in the given directory.
        
        :param train_data_dir: The directory to load training data from.
        :returns: None.  The model will be trained and saved.
        """
        self._load_data_from_dir(train_data_dir, train=True)
        self.model.fit(self.train_GLCM, self.train_classes_GLCM)
        pickle.dump(model, open(os.path.join(self.save_path, self.model_type.upper(), "GLCM_model.sav"), 'wb'))
 
    def load_model(self, path_to_model):
        """
        Load a trained model into this model from the given path.
        
        :param path_to_model: The path to the model save file.
        :returns:           None.  The model will be initialized with the saved
                            parameters.
        """
        self.model = pickle.load(open(path_to_model, 'rb'))

    def predict(self, img):
        """
        Predict the count for the given image.
        
        :param img: The image to predict for
        :returns: The number of plants in the image
        """
        
        return self.model.predict([img])
    
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
        
         
        return self.GLCM_feature(padded)



    def predict_generator(self, data_dir, num=0):
        """
        Predict the counts for all the images in the given data directory.
        
        :param data_dir: The directory to get the images from
        :returns: The predicted counts for all images in the given directory.
        """
        self._load_data_from_dir(data_dir, train=False)
        return self.predict(self.test_GLCM)

