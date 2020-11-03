"""
BRAIN CANCER - IS OR NOT?

Main steps to this task (with code)
Import photos taking into account train and test set
Create labels to each photo - to point which is yes which is no
Changing photo into numbers

? training and testing data in one, the same script?
Changing that into two files or leave as it is now
"""

# import libraries______________
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

import cv2 # to resize pictures
import os
from random import shuffle
from tqdm import tqdm
import pickle # for saving data

# Data_________________________
TRAIN_DIR = r"C:\Users\angel\AppData\Local\Programs\Python\Sieci VSCode\sieci\Projekt\INPUT\train"
TEST_DIR = r"C:\Users\angel\AppData\Local\Programs\Python\Sieci VSCode\sieci\Projekt\INPUT\test"
# CATEGORIES = ["yes", "no"]

BATCH_SIZE = 100
IMG_SIZE = 50
LR = 1e-3

# MODEL_NAME = 'PresenceOfCancer-{}-{}.model'.format(LR, '2conv-basic')

# # Processing the data
TRAIN = len(os.listdir(TRAIN_DIR)) # number of photos - train -> 202
TEST = len(os.listdir(TEST_DIR)) # number of photos - test -> 56
# print("Train: {}, Test: {}".format(TRAIN, TEST))

# get to computer know which photo is yes and which is no 
def label_img(img):
    word_label = img[0] # first letter define it
    if word_label == "Y": return [1, 0]
    elif word_label == "N": return [0, 1]

# _______Building data_______

# training data
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
print("Train:", len(training_data)) # 202
print(training_data[:3]) # [[array1, [0,1]], [array2, [0,1]] ... ]

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

# preparing data to a version which is available in neural network
X, y = [], []
for features, label in training_data:
    X.append(features)
    y.append(label)
 
# X should be a numpy array; -1 that could mean "any number"
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # 1 because it is a grayscale
y = np.array(y).reshape(-1, 1) # also reshape to be consistent with x

# Saving training data 
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# always is a possible to load it to our current script
# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)

print("X_0:", X[0])