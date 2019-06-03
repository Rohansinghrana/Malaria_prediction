# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:47:24 2019

@author: ROHAN
"""


#importing libraries
from pathlib import Path
import pandas as pd  
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


# Define path to the data directory
data_dir = Path('..\Malaria_proj\cell_images')

# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir/'train'
# Path to validation directory
test_dir = data_dir/'test'

#test
Uninfected_cases_test = test_dir/'Uninfected'
Parasitized_cases_test = test_dir/'Parasitized'

# Get the list of all the images for test
Uninfected_cases_test = Uninfected_cases_test.glob('*.png')
Parasitized_cases_test = Parasitized_cases_test.glob('*.png')

test_data=[]
test_s=[]
for img in Uninfected_cases_test:
    test_data.append((img,0))
    test_s.append((0))
# for malaria case and value of those is 1
for img in Parasitized_cases_test:
    test_data.append((img,1))
    test_s.append((1))
    
test_data = pd.DataFrame(test_data, columns=['image','label'])
test_s=pd.DataFrame(test_s,columns=['label'])
test_data = test_data.sample(frac=1.).reset_index(drop=True)

#Path to sub-director
Uninfected_cases_dir = train_dir/'Uninfected'
Parasitized_cases_dir = train_dir/'Parasitized'

# Get the list of all the images
Uninfected_cases = Uninfected_cases_dir.glob('*.png')
Parasitized_cases = Parasitized_cases_dir.glob('*.png')
# An empty list for  inserting new data
train_data = []
# for uninfected case and value of those is 0
for img in Uninfected_cases:
    train_data.append((img,0))
# for malaria case and value of those is 1
for img in Parasitized_cases:
    train_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'label'])

# Shuffle the data 
train_data = train_data.sample(frac=1.).reset_index(drop=True)

#length of images
n = len(train_data)


#creating a object
classifier = Sequential()
#convolution
classifier.add(Convolution2D(64,3,3, input_shape=(64,64,3),activation='relu'))
#pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#use flatten function
classifier.add(Flatten())
#full connection
classifier.add(Dense(output_dim=256,activation='relu'))#2nd layer
classifier.add(Dense(output_dim=128,activation='relu'))#3rd layer
classifier.add(Dense(output_dim=64,activation='relu'))#4th layer
classifier.add(Dense(output_dim=1,activation='sigmoid'))#final
#complie
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the CNN into the images

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#training set
train_set = train_datagen.flow_from_directory('cell_images/train' ,target_size=(64,64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('cell_images/test' ,target_size=(64,64),batch_size=32,class_mode='binary')

classifier.fit_generator(train_set,samples_per_epoch=n,nb_epoch=20,validation_data=test_set,nb_val_samples=16)

# For prediction
p = classifier.predict_generator(test_set) 
prediction = pd.DataFrame(p)
print("Prediction for the test 103 images dataset are:")
print('\n')

for i in range(0,205):
    if p[i]==0:
        print('Uninfected')
    else:
        print('Parasitized')
print(prediction)

#for the confusion matrix
# cm2 = confusion_matrix(prediction,test_s)
