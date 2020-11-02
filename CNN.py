"""
BRAIN CANCER - IS OR NOT?

Main steps to this task (with code)
TODO: Import photos taking into account train and test set - validation set (?)
TODO: Create labels to each photo - to point which is yes which is no
TODO: Changing photo into numbers
TODO: Conv + MaxPooling 
TODO: Predictions
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
TRAIN_DIR = r"C:\Users\angel\AppData\Local\Programs\Python\Sieci VSCode\sieci\Projekt\INPUT\train"
TEST_DIR = r"C:\Users\angel\AppData\Local\Programs\Python\Sieci VSCode\sieci\Projekt\INPUT\test"
CATEGORIES = ["yes", "no"]

BATCH_SIZE = 100
IMG_SIZE = 50
LR = 1e-3

# MODEL_NAME = 'PresenceOfCancer-{}-{}.model'.format(LR, '2conv-basic')


# # get to computer know which photo is yes and which is no 
# def label_img(img):
#     word_label = img[0] # first letter define it
#     if word_label == "Y": return [1, 0]
#     elif word_label == "N": return [0, 1]

# def create_train_data():
#     training_data = []
#     for img in tqdm(os.listdir(TRAIN_DIR)): # show a smart progress of loops 
#         label = label_img(img)
#         path = os.path.join(TRAIN_DIR, img)
#         img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
#         training_data.append([np.array(img), np.array(label)])
#     shuffle(training_data)    
#     np.save('train_data.npy', training_data)
#     return training_data


# # Processing the data
# TRAIN = len(os.listdir(TRAIN_DIR)) # number of photos - train
# TEST = len(os.listdir(TEST_DIR)) # number of photos - test

# img_train, img_test = [], []

# # import photos from train and test





