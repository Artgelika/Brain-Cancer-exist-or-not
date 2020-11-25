"""
BRAIN CANCER - IS OR NOT?

Main steps to this task (with code)
Import photos taking into account train and test set
Create labels to each photo - to point which is yes which is no
Changing photo into numbers

Changing that into two files or leave as it is now
"""

# import libraries______________
import numpy as np
np.random.seed(5)

import cv2 # to resize pictures
import os
from random import shuffle
from tqdm import tqdm
import pickle # for saving data
import random

# Data_________________________
TRAIN_DIR = r"C:\Users\angel\AppData\Local\Programs\Python\Sieci VSCode\sieci\Projekt\INPUT\train"
TEST_DIR = r"C:\Users\angel\AppData\Local\Programs\Python\Sieci VSCode\sieci\Projekt\INPUT\test"

BATCH_SIZE = 100
IMG_SIZE = 50

# # Processing the data
TRAIN = len(os.listdir(TRAIN_DIR)) # number of photos - train -> 202
TEST = len(os.listdir(TEST_DIR)) # number of photos - test -> 51
# print("Train: {}, Test: {}".format(TRAIN, TEST))


# get to computer know which photo is yes and which is no 
def label_img(img):
    word_label = img[0] # first letter define it
    if word_label == "Y": return 1 # [1, 0]
    elif word_label == "N": return 0 # [0, 1]


# _______Building data_______

### training data ###
training_data = []
def create_train_data():
    for img in tqdm(os.listdir(TRAIN_DIR)): # show a smart progress of loops 
        try:
            label = label_img(img)
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        except Exception as e:
            print("General exception", e)
    shuffle(training_data)    
    # np.save('train_data.npy', training_data)
    return training_data

create_train_data()
# print("Train:", len(training_data)) # 202
# print(training_data[:3]) # [[array1, array(0)], [array2, array(1)] ... ]

# here is the place to shuffle
random.shuffle(training_data)


# preparing data to a version which is available in neural network
X_train, y_train = [], []
for features, label in training_data:
    X_train.append(features)
    y_train.append(label)


# # X should be a numpy array; -1 that could mean "any number"
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # 1 because it is a grayscale
# y = np.array(y_train).reshape(-1, 2) # also reshape to be consistent with x

print("X train: {}, y train: {}".format(len(X_train), len(y_train)))
# print("Y test sth:", y[:3])

# Saving training data 
pickle_out = open("X.train", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y.train", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

# always is a possible to load it to our current script
# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)


### test data ###
# Similarly creating test data
testing_data = []
def create_test_data():
    for img in tqdm(os.listdir(TEST_DIR)): # show a smart progress of loops 
        try:
            label = label_img(img)
            path = os.path.join(TEST_DIR, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img), np.array(label)])
        except Exception as e:
            print("General exception", e)
    shuffle(testing_data)    
    # np.save('test_data.npy', testing_data)
    return testing_data

create_test_data()
# print("Test:", len(testing_data)) # 51
# print(testing_data[:3]) # [[array1, [0,1]], [array2, [0,1]] ... ]

random.shuffle(testing_data)

# preparing data to a version which is available in neural network
X_test, y_test = [], []
for features, label in testing_data:
    X_test.append(features)
    y_test.append(label)
 
# X should be a numpy array; -1 that could mean "any number"
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # 1 because it is a grayscale

print("X_test: {}, y_test: {}".format(len(X_test), len(y_test)))

# Saving training data 
pickle_out = open("X.test", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y.test", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


