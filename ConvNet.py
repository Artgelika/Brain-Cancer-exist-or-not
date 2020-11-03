"""
BRAIN CANCER - IS OR NOT?

Conv + MaxPooling 
Predictions

Data from X.pickle and y.pickle which  was generated in CNN.py file

# * I need to experiment with activation functions and number of epochs, because accuracy is low
"""

# import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
import pickle

# data
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

# import data
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
# print("X: ", X)
# print("y: ", y)
# y2 = []
# for el in y:
#     y2.append(list(el))
# print(y2[:10])
# Normalizing that data - scale that data
X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3,3), input_shape = X.shape[1:])) # 1: because we needn't to -1
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(256, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])
            
model.fit(X, y, BATCH_SIZE, EPOCHS, validation_split=0.1) # batch_size= epochs = 


