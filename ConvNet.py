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
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# data
BATCH_SIZE = 15
EPOCHS = 800
LR = 1e-3
VALIDATION = 0.1 # part of test data which will be a validation set: from 0 to 1


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


# creating two sets from training data
X_val = X_train[:int(len(X_train)*VALIDATION)]
y_val = y_train[:int(len(y_train)*VALIDATION)]

X_train = X_train[int(len(X_train)*VALIDATION):]
y_train = y_train[int(len(y_train)*VALIDATION):]


########## PLOT PRESENTING CLASSES LAYOUT ##########
# Number of examples
print("TRAIN: No:{}, Yes:{} ".format(list(y_train).count(0), list(y_train).count(1))) # -> 
print("VALIDATION: No:{}, Yes:{} ".format(list(y_val).count(0), list(y_val).count(1))) # -> 
print("TEST: No:{}, Yes:{} ".format(list(y_test).count(0), list(y_test).count(1))) # ->
Count = [[list(y_train).count(0), list(y_val).count(0), list(y_test).count(0)],  # no
        [list(y_train).count(1), list(y_val).count(1), list(y_test).count(1)]]   # yes

# visualize it - Bar Plot #####################
No = Count[0]
Yes = Count[1]
Set = ['Train set', 'Validation set', 'Test set']

X = np.arange(len(Set))
width = 0.35

fig, ax = plt.subplots()
no = ax.bar(X - width/2, No, width, color = 'b', label="No")
yes = ax.bar(X + width/2, Yes, width, color = 'g', label="Yes")
ax.set_ylabel('Count')
ax.set_xticks(X)
ax.set_xticklabels(Set)
ax.set_title('Count of classes in each set')
ax.legend()

def autolabel(tumor):
    """Attach a text label above each bar in *Set*, displaying its amount."""
    for rect in tumor:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
# 
autolabel(no)
autolabel(yes)

fig.tight_layout()

# plt.show()
###############################################


# Create a model
model = Sequential() # Sequential - the way to build a model in Keras layer by layer

model.add(Conv2D(48, (3,3), input_shape = X_train.shape[1:])) # 1: because we needn't to -1
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(16, (3,3), strides=(2,2), padding="valid"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(96, (3,3), strides=(2,2), padding="valid"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64)) 

model.add(Dense(1))
model.add(Activation('sigmoid'))

# Train the model
model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])


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


# to save the best model
checkpoint = keras.callbacks.ModelCheckpoint("model_checkpoint.hdf5", 
            monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='auto', period=1)


csv_logger = keras.callbacks.CSVLogger("Brain_tumor_detection.csv", separator=",", append=False)


early_stop = keras.callbacks.EarlyStopping(patience=30)

 
# Fit the model
history = model.fit(augment_data.flow(X_train, y_train, batch_size=BATCH_SIZE),
                              epochs = EPOCHS, validation_data=(X_val, y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                              callbacks=[learning_rate_reduction, csv_logger, checkpoint, early_stop])


# Model performance #######################
# plot model performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(history.epoch) + 1)

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Set')
plt.plot(epochs_range, val_acc, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Val Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')

plt.tight_layout()
plt.show()
##########################################


# predicting the test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/51)*100))

sns.heatmap(cm,annot=True)
plt.savefig('h.png')
plt.show()


