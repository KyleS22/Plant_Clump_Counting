
# coding: utf-8

# In[130]:


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

# # LBP and GLCM Helper Functions

# In[131]:


def GLCM_feature(Image):
            
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

def get_LBP(training_image):
    
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


# # Feature Extraction on Combined Downgraded Images

# In[132]:



#LBPH_features=[]
#classes=[]
#test_GLCM = []
#test_LBPH = []
#train_GLCM = []
#train_LBPH = []
#train_classes_GLCM = []
#train_classes_LBPH = []
#test_classes_GLCM =[]
#test_classes_LBPH =[]
#
#rotation_angles=[0,90,180,270]


class GLCMModel:

    def __init__(self, model_type=None, save_path=None):
        
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


    def _load_data_from_dir(self, path_to_data, train=True):
        for root, dirs, files in os.walk(path_to_data):
            for filename in files:
                cropped_image = sk.img_as_float(io.imread(root+'/'+filename))
                image_class = root.replace("\\","/").split('/')[-1]
                
                if train:
                    for angle in self.rotation_angles:
                        rotated_image = exp.rescale_intensity(rotate(cropped_image,angle))
                       # print("Training Image: "+str(filename)+" Class: "+str(image_class)+" Angel:"+str(angle))
                        self.train_GLCM.append(GLCM_feature(rotated_image))
                        self.train_classes_GLCM.append(int(image_class))
                else:
                    self.test_GLCM.append(GLCM_feature(cropped_image))
                    self.test_classes_GLCM.append(int(image_class))

       
    def fit(self, train_data_dir):
        self._load_data_from_dir(train_data_dir, train=True)
        self.model.fit(self.train_GLCM, self.train_classes_GLCM)
        pickle.dump(model, open(os.path.join(self.save_path, self.model_type.upper(), "GLCM_model.sav"), 'wb'))
 
    def load_model(self, path_to_model):
        self.model = pickle.load(open(path_to_model, 'rb'))

    def predict(self, data_dir):
        self._load_data_from_dir(data_dir, train=False)
        return self.model.predict(self.test_GLCM)

    def predict_generator(self, data_dir, num=0):
        return self.predict(data_dir)

class LBPHModel:

    def __init__(self, model_type=None, save_path=None):
        
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


    def _load_data_from_dir(self, path_to_data, train=True):
        for root, dirs, files in os.walk(path_to_data):
            for filename in files:
                cropped_image = sk.img_as_float(io.imread(root+'/'+filename))
                image_class = root.replace("\\","/").split('/')[-1]
                if train:
                    for angle in self.rotation_angles:
                        rotated_image = exp.rescale_intensity(rotate(cropped_image,angle))
                       # print("Training Image: "+str(filename)+" Class: "+str(image_class)+" Angel:"+str(angle))
                        self.train_LBPH.append(get_LBP(rotated_image))
                        self.train_classes_LBPH.append(int(image_class))
                else:
                    self.test_LBPH.append(get_LBP(cropped_image))
                    self.test_classes_LBPH.append(int(image_class))

       
    def fit(self, train_data_dir):
        self._load_data_from_dir(train_data_dir, train=True)
        self.model.fit(self.train_GLCM, self.train_classes_GLCM)
        pickle.dump(model, open(os.path.join(self.save_path, self.model_type.upper(), "LBPH_model.sav"), 'wb'))
 
    def load_model(self, path_to_model):
        self.model = pickle.load(open(path_to_model, 'rb'))

    def predict(self, data_dir):
        self._load_data_from_dir(data_dir, train=False)
        return self.model.predict(self.test_LBPH)

    def predict_generator(self, data_dir, num=0):
        return self.predict(data_dir)



#for root, dirs, files in os.walk('Data/combined_train_resized'):
#    for filename in files:
#        cropped_image = sk.img_as_float(io.imread(root+'/'+filename))
#        image_class = root.replace("\\","/").split('/')[2]
#        for angle in rotation_angles:
#            rotated_image = exp.rescale_intensity(rotate(cropped_image,angle))
#            print("Training Image: "+str(filename)+" Class: "+str(image_class)+" Angel:"+str(angle))
#            train_GLCM.append(GLCM_feature(rotated_image))
#            train_LBPH.append(get_LBP(rotated_image))
#            train_classes_GLCM.append(int(image_class))
#            train_classes_LBPH.append(int(image_class))
#
#
#for root, dirs, files in os.walk('Data/combined_val_resized/'):
#    for filename in files:
#        cropped_image =  sk.img_as_float(io.imread(root+'/'+filename))
#        image_class = root.replace("\\","/").split('/')[2]
#        resized_image = exp.rescale_intensity(cropped_image)
#        print("Testing Image: "+str(filename)+" Class: "+str(image_class))
#        test_GLCM.append(GLCM_feature(resized_image))
#        test_LBPH.append(get_LBP(resized_image))
#        test_classes_GLCM.append(int(image_class))
#        test_classes_LBPH.append(int(image_class))
#
#
## # Train Test Split
#
## In[133]:
#
#
## train_GLCM,test_GLCM,train_classes_GLCM,test_classes_GLCM=train_test_split(GLCM_features, classes, test_size=0.25)
## train_LBPH,test_LBPH,train_classes_LBPH,test_classes_LBPH=train_test_split(LBPH_features, classes, test_size=0.25)
#
#
## # KNN Classifier for Synthetic and Combined LBP and GLCM
#
## In[134]:
#
#
#model = KNeighborsClassifier(n_neighbors=10)
#model.fit(train_GLCM,train_classes_GLCM)
#pickle.dump(model, open('models/KNN_GLCM_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/KNN_GLCM_model.sav', 'rb'))
#predicted_classes = loaded_model.predict(test_GLCM)
#print("\nGLCM Confusion Matrix")
#print(confusion_matrix(test_classes_GLCM, predicted_classes))
#print("GLCM Accuracy Score")
#print(accuracy_score(test_classes_GLCM,predicted_classes))
#
#
#model = KNeighborsClassifier(n_neighbors=10)
#model.fit(train_LBPH,train_classes_LBPH)
#pickle.dump(model, open('models/KNN_LBPH_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/KNN_LBPH_model.sav', 'rb'))
#predicted_classes = loaded_model.predict(test_LBPH)
#print("\nLBPH Confusion Matrix")
#print(confusion_matrix(test_classes_LBPH, predicted_classes))
#print("LPBH Accuracy Score")
#print(accuracy_score(test_classes_LBPH,predicted_classes))
#
#
## # SVM Classifier for Synthetic and Combined LBP and GLCM
#
## In[135]:
#
#
#svm_rbf = svm.SVC(kernel='rbf', C=10)
#svm_rbf.fit(train_GLCM,train_classes_GLCM)
#pickle.dump(svm_rbf, open('models/SVM_GLCM_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/SVM_GLCM_model.sav', 'rb'))
#rbf_predicted_classes = loaded_model.predict(test_GLCM)
#print("\nGLCM Confusion Matrix")
#print(confusion_matrix(test_classes_GLCM, rbf_predicted_classes))
#print("GLCM Accuracy Score")
#print(accuracy_score(test_classes_GLCM,rbf_predicted_classes))
#
#
#svm_rbf = svm.SVC(kernel='rbf', C=10)
#svm_rbf.fit(train_LBPH,train_classes_LBPH)
#pickle.dump(svm_rbf, open('models/SVM_LBPH_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/SVM_LBPH_model.sav', 'rb'))
#rbf_predicted_classes = loaded_model.predict(test_LBPH)
#print("\nLBPH Confusion Matrix")
#print(confusion_matrix(test_classes_LBPH, rbf_predicted_classes))
#print("LPBH Accuracy Score")
#print(accuracy_score(test_classes_LBPH,rbf_predicted_classes))
#
#
## # GNB Classifier for Synthetic and Combined LBP and GLCM
#
## In[136]:
#
#
#gnb = GaussianNB()
#gnb.fit(train_GLCM,train_classes_GLCM)
#pickle.dump(gnb, open('models/GNB_GLCM_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/GNB_GLCM_model.sav', 'rb'))
#gnb_predicted_classes = loaded_model.predict(test_GLCM)
#print("\nGLCM Confusion Matrix")
#print(confusion_matrix(test_classes_GLCM, gnb_predicted_classes))
#print("GLCM Accuracy Score")
#print(accuracy_score(test_classes_GLCM,gnb_predicted_classes))
#
#gnb = GaussianNB()
#gnb.fit(train_LBPH,train_classes_LBPH)
#pickle.dump(gnb, open('models/GNB_LBPH_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/GNB_LBPH_model.sav', 'rb'))
#gnb_predicted_classes = loaded_model.predict(test_LBPH)
#print("\nLBPH Confusion Matrix")
#print(confusion_matrix(test_classes_LBPH, gnb_predicted_classes))
#print("LPBH Accuracy Score")
#print(accuracy_score(test_classes_LBPH,gnb_predicted_classes))
