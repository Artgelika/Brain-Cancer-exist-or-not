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


# data
BATCH_SIZE = 35
EPOCHS = 20
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
model = Sequential() # Sequential - the way to build a model in Keras layer by layer

model.add(Conv2D(256, (3,3), input_shape = X_train.shape[1:])) # 1: because we needn't to -1
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(256, (3,3), strides=(2,2), padding="valid"))
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
            optimizer="adam",
            metrics=['accuracy'])

# Define the Keras TensorBoard callback.
# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=LR)


# With data augmentation to prevent overfitting

augment_data = ImageDataGenerator( 
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

augment_data.fit(X_train)
augment_data.fit(X_val)
 
# Fit the model
history = model.fit(augment_data.flow(X_train, y_train, batch_size=BATCH_SIZE),
                              epochs = EPOCHS, validation_data=(X_val, y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                              callbacks=[learning_rate_reduction]) # , tensorboard

# visualize accuracy and loss

# four plots together_____________________

_, axs = plt.subplots(2,2)
axs[0,0].plot(history.history['loss'], 'r') # train loss
axs[0,0].set_ylabel("Error")
axs[0,0].set_xlabel("Epochs")
axs[0,0].set_title("loss")
axs[0,0].grid()

axs[0,1].plot(history.history['accuracy'], 'g')  # val loss
axs[0,1].set_ylabel("Accuracy")
axs[0,1].set_xlabel("Epochs")
axs[0,1].set_title("accuracy")
axs[0,1].grid()

axs[1,0].plot(history.history['val_loss'], 'r') # train loss
axs[1,0].set_ylabel("Error")
axs[1,0].set_xlabel("Epochs")
axs[1,0].set_title("val loss")
axs[1,0].grid()

axs[1,1].plot(history.history['val_accuracy'], 'g') # val loss
axs[1,1].set_ylabel("Accuracy")
axs[1,1].set_xlabel("Epochs")
axs[1,1].set_title("val accuracy")
axs[1,1].grid()

plt.subplots_adjust(wspace = 0.5, hspace=0.5)
plt.savefig('accuracy_loss_four.png')
plt.show()
# ________________________

# two plots in one

# 1 loss
plt.close()
plt.plot(history.history['val_loss'], 'r', history.history['loss'], 'b')
plt.grid()
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["val_loss", "train_loss"])
plt.savefig('val_train_loss.png')
plt.show()

# 2 acc
plt.close()
plt.plot(history.history['val_accuracy'], 'r', history.history['accuracy'], 'b')
plt.grid()
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["val_acc", "train_acc"])
plt.savefig('val_train_acc.png')
plt.show()

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


