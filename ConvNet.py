"""
BRAIN CANCER - IS OR NOT?

Conv + MaxPooling 
Predictions

Data from X.pickle and y.pickle which  was generated in CNN.py file

"""
# * with current settings, acuuracy is approx 85,3%

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
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# data
BATCH_SIZE = 35
EPOCHS = 16
LR = 1e-4
VALIDATION = 0.1 # part of test data which will be a validation set: from 0 to 1

# MODEL_NAME = 'PresenceOfCancer-{}-{}.model'.format(LR, '2conv-basic')
# tensorboard = TensorBoard(log_dir='logs/{}'.format(MODEL_NAME))

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX") # easy way to notice where is beginning of specific compilation
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
model = Sequential() # Sequential - the way to build a model in Keras layer by layer

model.add(Conv2D(256, (3,3), input_shape = X_train.shape[1:])) # 1: because we needn't to -1
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


#########################################
## I've tried to use another combination of model
# model = Sequential() # Sequential - the way to build a model in Keras layer by layer

# # model = Sequential() # Sequential - the way to build a model in Keras layer by layer

# model.add(Conv2D(256, (3,3), input_shape = X_train.shape[1:])) # 1: because we needn't to -1
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(256, (3,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation("relu"))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))
####################################################

# # Train the model
model.compile(loss="binary_crossentropy", # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=LR)


# With data augmentation to prevent overfitting

datagen = ImageDataGenerator( 
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
datagen.fit(X_val)

# Fit the model
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                              epochs = EPOCHS, validation_data=(X_val, y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                              callbacks=[learning_rate_reduction])

# model.summary()

###
# # predicting the test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/51)*100))

sns.heatmap(cm,annot=True)
plt.savefig('h.png')
plt.show()
