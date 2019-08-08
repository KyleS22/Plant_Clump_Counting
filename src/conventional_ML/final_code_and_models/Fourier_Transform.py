
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


region_features=[]
classes=[]
test_region = []
train_region = []
train_classes_region = []
test_classes_region =[]

rotation_angles=[0,90,180,270]

minimum=0.0
maximum=0.0

for root, dirs, files in os.walk('Data/combined_train_resized'):
    for filename in files:
        cropped_image =  io.imread(root+'/'+filename)
        cropped_image = filt.unsharp_mask(cropped_image, radius=1, amount=4)
        image_class = root.replace("\\","/").split('/')[2]
        for angle in rotation_angles:
            print("Training Image: "+str(filename)+" Class: "+str(image_class)+" Angle: "+str(angle))
            fourier = fft(cropped_image).flatten()
            fourier = np.true_divide(fourier, np.count_nonzero(fourier))
            hist,bins = np.histogram(fourier,bins=5)
            train_region.append(hist)
            train_classes_region.append(int(image_class))
            
for root, dirs, files in os.walk('Data/combined_val_resized'):
    for filename in files:
        cropped_image =  io.imread(root+'/'+filename)
        cropped_image = filt.unsharp_mask(cropped_image, radius=1, amount=4)
        image_class = root.replace("\\","/").split('/')[2]
        for angle in rotation_angles:
            print("Testing Image: "+str(filename)+" Class: "+str(image_class)+" Angle: "+str(angle))
            fourier = fft(cropped_image).flatten()
            fourier = np.true_divide(fourier, np.count_nonzero(fourier))
            hist,bins = np.histogram(fourier,bins=5)
            test_region.append(hist)
            test_classes_region.append(int(image_class))


# # Classification

# In[16]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_region,train_classes_region)
pickle.dump(model, open('models/KNN_FFT_model.sav', 'wb'))
loaded_model = pickle.load(open('models/KNN_FFT_model.sav', 'rb'))
predicted_classes = loaded_model.predict(test_region)
print("\nKNN Confusion Matrix")
print(confusion_matrix(test_classes_region, predicted_classes))
print("KNN Accuracy Score")
print(accuracy_score(test_classes_region,predicted_classes))


svm_rbf = svm.SVC(kernel='rbf', C=50)
svm_rbf.fit(train_region,train_classes_region)
pickle.dump(svm_rbf, open('models/SVM_FFT_model.sav', 'wb'))
loaded_model = pickle.load(open('models/SVM_FFT_model.sav', 'rb'))
rbf_predicted_classes = loaded_model.predict(test_region)
print("\nSVM Confusion Matrix")
print(confusion_matrix(test_classes_region, rbf_predicted_classes))
print("SVM Accuracy Score")
print(accuracy_score(test_classes_region,rbf_predicted_classes))



gnb = GaussianNB()
gnb.fit(train_region,train_classes_region)
pickle.dump(gnb, open('models/GNB_FFT_model.sav', 'wb'))
loaded_model = pickle.load(open('models/GNB_FFT_model.sav', 'rb'))
gnb_predicted_classes = loaded_model.predict(test_region)
print("\n\nGNB Confusion Matrix")
print(confusion_matrix(test_classes_region, gnb_predicted_classes))
print("GNB Accuracy Score")
print(accuracy_score(test_classes_region,gnb_predicted_classes))

