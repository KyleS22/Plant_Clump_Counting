
# coding: utf-8

# In[12]:


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


# # Loading Images for FFT

# In[6]:

class FourierTransformModel:

    def __init__(self, model_type=None, save_path=None):
        
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
        self._load_data_from_dir(train_data_dir, train=True)
        self.model.fit(self.train_region, self.train_classes_region)
        pickle.dump(model, open(os.path.join(self.save_path, self.model_type.upper(), "FFT_model.sav"), 'wb'))
 
    def load_model(self, path_to_model):
        print("LOADING MODEL")
        self.model = pickle.load(open(path_to_model, 'rb'))
    
    def predict(self, data_dir):
        self._load_data_from_dir(data_dir, train=False)
        return self.model.predict(self.test_region)

    def predict_generator(self, data_dir, num=0):
        return self.predict(data_dir)

#for root, dirs, files in os.walk('Data/combined_val_resized'):
#    for filename in files:
#        cropped_image =  io.imread(root+'/'+filename)
#        cropped_image = filt.unsharp_mask(cropped_image, radius=1, amount=4)
#        image_class = root.replace("\\","/").split('/')[2]
#        for angle in rotation_angles:
#            print("Testing Image: "+str(filename)+" Class: "+str(image_class)+" Angle: "+str(angle))
#            fourier = fft(cropped_image).flatten()
#            fourier = np.true_divide(fourier, np.count_nonzero(fourier))
#            hist,bins = np.histogram(fourier,bins=5)
#            test_region.append(hist)
#            test_classes_region.append(int(image_class))
#
#
# # Classification

# In[16]:


#model = KNeighborsClassifier(n_neighbors=5)
#model.fit(train_region,train_classes_region)
#pickle.dump(model, open('models/KNN_FFT_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/KNN_FFT_model.sav', 'rb'))
#predicted_classes = loaded_model.predict(test_region)
#print("\nKNN Confusion Matrix")
#print(confusion_matrix(test_classes_region, predicted_classes))
#print("KNN Accuracy Score")
#print(accuracy_score(test_classes_region,predicted_classes))


#svm_rbf = svm.SVC(kernel='rbf', C=50)
#svm_rbf.fit(train_region,train_classes_region)
#pickle.dump(svm_rbf, open('models/SVM_FFT_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/SVM_FFT_model.sav', 'rb'))
#rbf_predicted_classes = loaded_model.predict(test_region)
#print("\nSVM Confusion Matrix")
#print(confusion_matrix(test_classes_region, rbf_predicted_classes))
#print("SVM Accuracy Score")
#print(accuracy_score(test_classes_region,rbf_predicted_classes))



#gnb = GaussianNB()
#gnb.fit(train_region,train_classes_region)
#pickle.dump(gnb, open('models/GNB_FFT_model.sav', 'wb'))
#loaded_model = pickle.load(open('models/GNB_FFT_model.sav', 'rb'))
#gnb_predicted_classes = loaded_model.predict(test_region)
#print("\n\nGNB Confusion Matrix")
#print(confusion_matrix(test_classes_region, gnb_predicted_classes))
#print("GNB Accuracy Score")
#print(accuracy_score(test_classes_region,gnb_predicted_classes))

