
# coding: utf-8

# In[1]:


import pickle
import gzip
from PIL import Image
import os
import numpy as np
import math
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# %matplotlib inline
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from tqdm import tqdm_notebook
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix


# In[2]:


#Training Percentage
training_percent = 80
#Validation Percentage
validation_percent = 10
#Test Percentage
test_percent = 10


# In[3]:


def convert_to_one_hot_vector(input_vector, hot_vector_size=10):
    target_vector = [[] for i in range(len(input_vector))]
    for i in range(len(input_vector)):
        target_vector[i] = [1 if j == input_vector[i] else 0 for j in range(hot_vector_size)]
    return target_vector


# In[4]:


def generate_training_data(feature_set, training_percent = 80):
    len_training = int(math.ceil(len(feature_set)*(training_percent*0.01)))
    t = feature_set[:len_training]
    return t


# In[5]:


def generate_testing_data(feature_set, count_training, testing_percent = 10):
    len_training = int(math.ceil(len(feature_set)*(testing_percent*0.01)))
    end_training = count_training + len_training
    t = feature_set[count_training + 1:end_training]
    return t


# In[6]:


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
MNIST_training_data, MNIST_validation_data, MNIST_test_data = pickle.load(f, encoding='latin1')
f.close()


# In[7]:


MNIST_training_target_vector = convert_to_one_hot_vector(MNIST_training_data[1])
MNIST_validation_target_vector = convert_to_one_hot_vector(MNIST_validation_data[1])
MNIST_test_target_vector = convert_to_one_hot_vector(MNIST_test_data[1])


# In[8]:


USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


# In[9]:


USPS_data = np.asarray(USPSMat)
USPS_target_vector = convert_to_one_hot_vector(np.asarray(USPSTar))


# In[10]:


def softmax(a):
    a_max = np.max(a, axis=1, keepdims=True)
    a = np.exp(a - a_max)
    return a / np.sum(a, axis=1, keepdims=True)
#     return np.asarray((np.exp(a.T) / np.sum(np.exp(a), axis=1)).T)

def get_val_test(feature_set,W):
    Y = np.dot(feature_set,W)
    return Y

def get_accuracy(RegressionTarget, ExistingTarget):
    accuracy = 0.0

    for i in range(len(RegressionTarget)):
        RegressionTarget[i] = [1 if j == max(RegressionTarget[i]) else 0 for j in RegressionTarget[i]]
    
    matchedTarget = 0
    
    for i in range(len(ExistingTarget)):
        if(np.array_equal(RegressionTarget[i],ExistingTarget[i])):
            matchedTarget += 1
    
    if(matchedTarget > 0):
        accuracy = (matchedTarget/len(ExistingTarget))*100
    return accuracy


# In[12]:


training_phi = MNIST_training_data[0]
validation_phi = MNIST_validation_data[0]
testing_phi = MNIST_test_data[0]
usps_phi = USPS_data

number_of_classes = 10
training_data = MNIST_training_data[0]
weights = np.random.rand(784,10)
bias = np.random.rand(1,10)
epochs = 20
batch_size = 16
lamb = 0.03
learning_rate = 0.025
losses = []
L_Accuracy_TR = []
L_Accuracy_Val = []
L_Accuracy_Test = []

USPS_Accuracy = []

for epochCount in tqdm_notebook(range(epochs)):
    for i in range(int(len(training_data)/batch_size)):
        batch = training_data[batch_size*i:batch_size*i + batch_size]
        batch_target = np.asarray(MNIST_training_target_vector[batch_size*i:batch_size*i + batch_size])
        m = len(batch)
        y = np.dot(batch,weights) + bias
        activated_y = softmax(y)
        loss = (-1 / m) * np.sum(batch_target * np.log(activated_y)) + (lamb/2)*np.sum(weights*weights)
        losses.append(loss)
        gradient = (-1 / m) * np.dot(batch.T, (batch_target - activated_y)) + lamb*weights
        
        weights = weights - (learning_rate * gradient)

TR_TEST_OUT   = get_val_test(training_phi,weights)
L_Accuracy_TR.append(get_accuracy(TR_TEST_OUT,MNIST_training_target_vector))

VAL_TEST_OUT = get_val_test(validation_phi,weights)
L_Accuracy_Val.append(get_accuracy(VAL_TEST_OUT,MNIST_validation_target_vector))

TEST_OUT = get_val_test(testing_phi,weights)
L_Accuracy_Test.append(get_accuracy(TEST_OUT,MNIST_test_target_vector))

USPS_TEST_OUT = get_val_test(usps_phi,weights)
USPS_Accuracy.append(get_accuracy(USPS_TEST_OUT,USPS_target_vector))
    

print ("-------------------------- LOGISTIC REGRESSION ------------------------------")
print ("-------------------------------- MNIST --------------------------------------")
print ("Accuracy Training   = " + str(np.around(L_Accuracy_TR[len(L_Accuracy_TR)-1],5)))
print ("Accuracy Validation = " + str(np.around(L_Accuracy_Val[len(L_Accuracy_Val)-1],5)))
print ("Accuracy Testing    = " + str(np.around(L_Accuracy_Test[len(L_Accuracy_Test)-1],5)))

print ("-------------------------- LOGISTIC REGRESSION ------------------------------")
print ("--------------------------------- USPS --------------------------------------")
print ("Accuracy Training   = " + str(np.around(USPS_Accuracy[len(USPS_Accuracy)-1],5)))

LOG_MNIST_confucio = confusion_matrix(np.argmax(MNIST_test_target_vector, axis=1), np.argmax(TEST_OUT, axis=1))
LOG_USPS_confucio = confusion_matrix(np.argmax(USPS_target_vector, axis=1),np.argmax(USPS_TEST_OUT, axis=1))

print ("MNIST Confusion Matrix \n" + str(LOG_MNIST_confucio))
print ("USPS Confusion Matrix \n" + str(LOG_USPS_confucio))


# In[ ]:


# plt.rcParams['figure.figsize'] = (10, 8)
# plt.plot(losses)


# In[ ]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes=10
image_vector_size=28*28

x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
image_size = 784

model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=False,validation_split=.1)


# In[ ]:


MNIST_predicted_target =  np.argmax(model.predict(x_test, verbose =1 ), axis=1)
USPS_predicted_target =  np.argmax(model.predict(USPS_data, verbose =1 ), axis=1)
MNIST_loss,MNIST_accuracy = model.evaluate(x_test, y_test, verbose=False)
USPS_loss,USPS_accuracy = model.evaluate(USPS_data, np.array(USPS_target_vector), verbose=False)

NN_MNIST_confucio = confusion_matrix(np.argmax(y_test, axis=1), (MNIST_predicted_target))
NN_USPS_confucio = confusion_matrix(np.argmax(USPS_target_vector, axis=1),USPS_predicted_target)


# In[ ]:


print ("-------------------------- NEURAL NETWORK ------------------------------")
print ("MNIST Accuracy = " + str(MNIST_accuracy))
print ("USPS Accuracy = " + str(USPS_accuracy))
print ("MNIST Confusion Matrix \n" + str(NN_MNIST_confucio))
print ("USPS Confusion Matrix \n" + str(NN_USPS_confucio))


# In[ ]:


SVM_MNIST_training_data = MNIST_training_data[0]
SVM_MNIST_training_target_vector = np.argmax(MNIST_training_target_vector, axis =1)

SVM_MNIST_validation_data = MNIST_validation_data[0]
SVM_MNIST_validation_target_vector = np.argmax(MNIST_validation_target_vector, axis =1)

SVM_MNIST_test_data = MNIST_test_data[0]
SVM_MNIST_test_target_vector = np.argmax(MNIST_test_target_vector, axis =1)

# SVM
classifier1 = SVC(kernel='linear');
classifier1.fit(SVM_MNIST_training_data, SVM_MNIST_training_target_vector)
MNIST_validation_accuracy1 = classifier1.predict(SVM_MNIST_validation_data)
MNIST_test_accuracy1 = classifier1.predict(SVM_MNIST_test_data)
USPS_accuracy1 = classifier1.predict(USPS_data)

# SVM
classifier2 = SVC(kernel='rbf', gamma = 1);
classifier2.fit(SVM_MNIST_training_data, SVM_MNIST_training_target_vector)
MNIST_validation_accuracy2 = classifier2.predict(SVM_MNIST_validation_data)
MNIST_test_accuracy2 = classifier2.predict(SVM_MNIST_test_data)
USPS_accuracy2 = classifier2.predict(USPS_data)

# SVM
classifier3 = SVC(kernel='rbf');
classifier3.fit(SVM_MNIST_training_data, SVM_MNIST_training_target_vector)
MNIST_validation_accuracy3 = classifier3.predict(SVM_MNIST_validation_data)
MNIST_test_accuracy3 = classifier3.predict(SVM_MNIST_test_data)
USPS_accuracy3 = classifier3.predict(USPS_data)

print ("-------------------------- SVM ------------------------------")
print ("-------------------------- Linear ------------------------------")
print ("MNIST Validation Accuracy = " + str(MNIST_validation_accuracy1))
print ("MNIST Test Accuracy = " + str(MNIST_test_accuracy1))
print ("USPS Accuracy = " + str(USPS_accuracy1))
print ("-------------------------- RBF with Gamma = 1 ------------------------------")
print ("MNIST Validation Accuracy = " + str(MNIST_validation_accuracy2))
print ("MNIST Test Accuracy = " + str(MNIST_test_accuracy2))
print ("USPS Accuracy = " + str(USPS_accuracy2))
print ("-------------------------- RBF and Default values ------------------------------")
print ("MNIST Validation Accuracy = " + str(MNIST_validation_accuracy3))
print ("MNIST Test Accuracy = " + str(MNIST_test_accuracy3))
print ("USPS Accuracy = " + str(USPS_accuracy3))


# In[ ]:


RFC_MNIST_training_data = MNIST_training_data[0]
RFC_MNIST_training_target_vector = np.argmax(MNIST_training_target_vector, axis =1)

RFC_MNIST_validation_data = MNIST_validation_data[0]
RFC_MNIST_validation_target_vector = np.argmax(MNIST_validation_target_vector, axis =1)

RFC_MNIST_test_data = MNIST_test_data[0]
RFC_MNIST_test_target_vector = np.argmax(MNIST_test_target_vector, axis =1)

RFC_USPS_data = USPS_data

#RandomForestClassifier
RFC_classifier1 = RandomForestClassifier(n_estimators=100);
RFC_classifier1.fit(RFC_MNIST_training_data, RFC_MNIST_training_target_vector)
RFC_MNIST_validation_accuracy1 = RFC_classifier1.predict(RFC_MNIST_test_data)
RFC_USPS_test_accuracy1 = RFC_classifier1.predict(USPS_data)


# In[27]:


RFC_MNIST_confucio = confusion_matrix(np.argmax(MNIST_test_target_vector, axis=1), (RFC_MNIST_validation_accuracy1))
RFC_USPS_confucio = confusion_matrix(np.argmax(USPS_target_vector, axis=1),RFC_USPS_test_accuracy1)

print ("-------------------------- RFC ------------------------------")
print ("MNIST Test Accuracy = " + str((np.mean(np.argmax(MNIST_test_target_vector, axis=1) == RFC_MNIST_validation_accuracy1))*100))
print ("USPS Accuracy = " + str((np.mean(np.argmax(USPS_target_vector, axis=1) == RFC_USPS_test_accuracy1))*100))
print ("MNIST Confusion Matrix \n" + str(RFC_MNIST_confucio))
print ("USPS Confusion Matrix \n" + str(RFC_USPS_confucio))

