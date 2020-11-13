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

# Create a model
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
