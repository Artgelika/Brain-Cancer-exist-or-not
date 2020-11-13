"""
BRAIN CANCER - IS OR NOT?

Conv + MaxPooling 
Predictions

Data from X.pickle and y.pickle which  was generated in CNN.py file

"""

# import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# data
BATCH_SIZE = 35
EPOCHS = 16
LR = 1e-3
VALIDATION = 0.1 # part of test data which will be a validation set: from 0 to 1

MODEL_NAME = 'PresenceOfCancer-{}-{}.model'.format(LR, '2conv-basic')
tensorboard = TensorBoard(log_dir='logs/{}'.format(MODEL_NAME))

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX") # easily way to notice where is beginning of specific compilation
# import data
X = pickle.load(open("X.train","rb"))
y = pickle.load(open("y.train","rb"))

X_test = pickle.load(open("X.test","rb"))
y_test = pickle.load(open("y.test","rb"))


# Normalizing that data - scale that data
X = X/255.0
y = np.array(y).reshape(-1, 1)

# if I want to to use the above validation, I have to reduce the dimension to twice lower
# in this case I will take into consideration only first number from a label eg. [1,0]

# reshaping y test set - [0,1] is count as 1 element, not two

# creating two sets from training data
# without dividing below it works with second "fit" 
X_val = X[:int(len(X)*VALIDATION)]
y_val = X[:int(len(y)*VALIDATION)]

X_train = X[int(len(X)*VALIDATION):]
y_train = X[int(len(y)*VALIDATION):]

# ____shapes of matrices____
# print("X.shape", X.shape) # (202, 100, 100, 1)
# print("y.shape", y.shape) # (202, 2)

# print("X_val.shape", X_val.shape) # (20, 100, 100, 1)
# print("y_val.shape", y_val.shape) # (20, 2)

# print("X_train.shape", X_train.shape) # (182, 100, 100, 1)
# print("y_train.shape", y_train.shape) # (182, 2)

# ____quick look at how data is looking____
# print("X_train:", X_train[:3])
# print("y_train:", y_train[:3])

# print("X_val: ", X_val[:3])
# print("y_val: ", y_val[:3])

# print("X: ", X)
# print("y: ", y)


model = Sequential() # Sequential - the way to build a model in Keras layer by layer

model.add(Conv2D(256, (3,3), input_shape = X.shape[1:])) # 1: because we needn't to -1
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(256, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
# model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))


# Train the model
model.compile(loss="binary_crossentropy", # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=['accuracy'])


# print(y_train[:3])
# print("X:", len(X)) # 202
# print("Y:", len(y)) # 202
# print("X_train:", len(X_train)) # 182
# print("y_train:", len(y_train)) # 182
# print("X_val:", len(X_val)) # 20
# print("y_val:", len(y_val)) # 20 


# Thing below: "validation_data=(X_val, y_val)" cause an error because:  ValueError: logits and labels must have the same shape ((None, 1) vs (None, 2))
# history = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,  # validation_data=(X_val.T, y_val.T)
#                     validation_data=(X_val, y_val), shuffle=True) # ? transpose or not transpose?
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION, callbacks=[tensorboard]) # batch_size= epochs = 
# model.summary()

###
# # predicting the test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))

sns.heatmap(cm,annot=True)
plt.savefig('h.png')
plt.show()
