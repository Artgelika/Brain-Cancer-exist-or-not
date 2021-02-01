"""
BRAIN CANCER - IS OR NOT?

Conv + MaxPooling 
Predictions

Data from X.pickle and y.pickle which  was generated in CNN.py file

"""

# import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
# keras tuner
from kerastuner.applications import HyperResNet
from kerastuner.tuners import Hyperband
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import time


# data
BATCH_SIZE = 15 #100
EPOCHS = 300#2#000 # 603
LR = 1e-3
VALIDATION = 0.1 # part of test data which will be a validation set: from 0 to 1

# MODEL_NAME = 'PresenceOfCancer-{}-{}.model'.format(LR, '2conv-basic') # if I could use tensorboard
# tensorboard = TensorBoard(log_dir='logs/{}'.format(MODEL_NAME))

# import data
X_train = pickle.load(open("X.train","rb"))
y_train = pickle.load(open("y.train","rb"))

X_test = pickle.load(open("X.test","rb"))
y_test = pickle.load(open("y.test","rb"))


# Normalizing that data - scale that data
X_train = X_train/255.0
y_train = np.array(y_train).reshape(-1, 1)

# Normalizing test data - scale that data
X_test = X_test/255.0
y_test = np.array(y_test).reshape(-1, 1)


# # creating two sets from training data
X_val = X_train[:int(len(X_train)*VALIDATION)]
y_val = y_train[:int(len(y_train)*VALIDATION)]

X_train = X_train[int(len(X_train)*VALIDATION):]
y_train = y_train[int(len(y_train)*VALIDATION):]

# Create a model
def build_model(hp):
    model = Sequential() # Sequential - the way to build a model in Keras layer by layer

    model.add(Conv2D(hp.Int('input_units',
                             min_value=48,
                             max_value=240,
                             step=64), (3,3), input_shape = X_train.shape[1:])) # 1: because we needn't to -1
                              
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))

#     for _ in range(hp.Int('n_layers', 1, 3)):
#         model.add(Conv2D(hp.Int('input_units',
#                              min_value=16,
#                              max_value=128,
#                              step=32), (3,3)))
#         model.add(Activation("relu"))

    model.add(Conv2D(hp.Int('input_units',
                             min_value=16,
                             max_value=32,
                             step=16), (3,3), strides=(2,2), padding="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(hp.Int('input_units',
                             min_value=96,
                             max_value=128,
                             step=32), (3,3), strides=(2,2), padding="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())
    model.add(Dense(64)) # , kernel_initializer='uniform'
    # model.add(Activation("relu"))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
# opt = keras.optimizers.Adam(learning_rate=LR) # using this - the score is approx 60%

# # Train the model
    model.compile(loss="binary_crossentropy", # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="nadam",
                metrics=['accuracy'])
    return model

LOG_DIR = f"{int(time.time())}"

# first define model
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=1,  # how many model variations to test?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
    directory=LOG_DIR)

tuner.search_space_summary()

# add the search
tuner.search(x=X_train,
             y=y_train,
             verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
             epochs=EPOCHS,
             batch_size=BATCH_SIZE,
             #callbacks=[tensorboard],  # if you have callbacks like tensorboard, they go here.
             validation_data=(X_test, y_test))

tuner.results_summary()

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)


# tuner = pickle.load(open("tuner_1576628824.pkl","rb"))
tuner.get_best_hyperparameters()[0].values

tuner.get_best_models()[0].summary()
# tuner.results_summary()

