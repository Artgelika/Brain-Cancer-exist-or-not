# -*- coding: utf-8 -*-
"""
-iteruje po zdjeciach
TODO: Dowiedziec sie dlaczego w wielu przykladach z internetu
w CNN ludzie zamiast zdjec importuja csv ?
Moze by bardzo przeanalizowac ten przyklad, to sie dowiem.
@author: angel
"""

# import libraries______________
from PIL import Image
from numpy import asarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import cv2 # to resize pictures
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Data_________________________
TRAIN_DIR = r"C:\Users\angel\OneDrive\Pulpit\sieci\Projekt\INPUT\train"
TEST_DIR = r"C:\Users\angel\OneDrive\Pulpit\sieci\Projekt\INPUT\test"
IMG_SIZE = 50
LR = 1e-3

TRAIN = len(os.listdir(TRAIN_DIR)) # number of photos - train
TEST = len(os.listdir(TEST_DIR)) # number of photos - test

img_train, img_test = [], []

# import photos from train i test
# Y -> [1,0] N -> [0,1]
####################### TRAIN #######################
# TRAIN - YES
for i in range(TRAIN):
    try:
        # load the image
        filename = os.path.join(TRAIN_DIR, "Y" + str(i) + ".jpg")
        img = Image.open(filename)
        # img.show()
        # convert image to numpy array
        data = asarray(img)
        img_train.append(data)
    except:
        pass


# TRAIN - NO  
for i in range(TRAIN):
    try:
        # load the image
        filename = os.path.join(TRAIN_DIR, "N" + str(i) + ".jpg")
        img = Image.open(filename)
        # img.show()
        # convert image to numpy array
        data = asarray(img)
        img_train.append(data)
    except:
        pass
    
print(len(img_train))
print(img_train[0][0][0])

# Validation data

####################### TEST #######################
# TEST - YES
for i in range(TEST):
    try:
        # load the image
        filename = os.path.join(TRAIN_DIR, "Y" + str(i) + ".jpg")
        img = Image.open(filename)
        # img.show()
        # convert image to numpy array
        data = asarray(img)
        img_test.append(data)
    except:
        pass
    
# TEST - NO
for i in range(TEST):
    try:
        # load the image
        filename = os.path.join(TRAIN_DIR, "N" + str(i) + ".jpg")
        img = Image.open(filename)
        # img.show()
        # convert image to numpy array
        data = asarray(img)
        img_test.append(data)
    except:
        pass
    
print(len(img_test))
print(img_test[0][0][0])

# get to computer know which photo is yes and which no  
def label_img(img):
    word_label = img.split(".")[0][0]
    if word_label == "Y": return 1
    if word_label == "N": return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




