"""
BRAIN CANCER - IS OR NOT?

Conv + MaxPooling 
Predictions

Data from X.pickle and y.pickle which  was generated in CNN.py file

"""

# import libraries
# import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.activations import relu, elu, softmax
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
# from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import talos # to improve hyperparameters
from talos.utils import lr_normalizer


# data
# BATCH_SIZE = 35
VALIDATION = 0.1 # that must be constant because of dividing train data
# part of test data which will be a validation set: from 0 to 1
LIMIT = 200 # limit of rounds - talos

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
def model_brain_tumor(X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, p: dict):

    model = Sequential() # Sequential - the way to build a model in Keras layer by layer

#     model.add(Dense(, input_dim = X_train.shape[1])) # 
    model.add(Conv2D(p["first_neuron"], (3,3), input_shape = X_train.shape[1:])) # 1: because we needn't to -1
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(256, (3,3), strides=(2,2), padding="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())
    model.add(Dense(64)) # , kernel_initializer='uniform'
    # model.add(Activation("relu"))

    model.add(Dense(1))
    model.add(Activation(p['last_activation']))

    # opt = keras.optimizers.Adam(learning_rate=LR) # using this - the score is approx 60%

    # # Train the model
    model.compile(loss="binary_crossentropy", # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=p["optimizer"](lr=lr_normalizer(p['LR'], p['optimizer'])),
            metrics=['accuracy'])
     # Define the Keras TensorBoard callback.
     # logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
     # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=p['LR'])


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
    # augment_data.fit(X_val)
       
    # Fit the model
    history = model.fit(augment_data.flow(X_train, y_train, batch_size=p['BATCH_SIZE']), # augment_data.flow(X_train, y_train, batch_size=p['BATCH_SIZE']),
                            epochs = p['EPOCHS'], validation_data=(X_val, y_val),
                            verbose = 1, steps_per_epoch=X_train.shape[0] // p['BATCH_SIZE'],
                            callbacks=[learning_rate_reduction]) # , tensorboard

   
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    # making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    test_score = ((cm[0][0] + cm[1][1])/51)*100
    print("Our accuracy is {}%".format(test_score))
    

    sns.heatmap(cm,annot=True)
#     plt.savefig('h.png')
#     plt.show()
    
    return history, model
# using talos

p = {   "EPOCHS": [15,20,25,30,35],
        "BATCH_SIZE":[20, 25, 30, 35],
        "LR": [1e-1],
        "first_neuron": [256, 512],
        "dropout": [0, 0.3, 0.5, 0.7],
        "optimizer": [Nadam],
        "last_activation": ["sigmoid"]}

scan_object = talos.Scan(x=X_train, y=y_train, model=model_brain_tumor, params=p, experiment_name='brain_tumor', round_limit=LIMIT)


# accessing the results data frame
"Scan data head(): ",scan_object.data.head()

# accessing epoch entropy values for each round
"learning_entropy: ", scan_object.learning_entropy

# access the summary details
"details:", scan_object.details

# accessing the saved models
"saved_models: ", scan_object.saved_models

# accessing the saved weights for models
"saved_weights: ",scan_object.saved_weights

# use Scan object as input
analyze_object = talos.Analyze(scan_object)
# access the dataframe with the results
"analyze_object: ", analyze_object.data
# get the number of rounds in the Scan
"analyze rounds: ", analyze_object.rounds()

# get the highest result for any metric
"Analyze val: ", analyze_object.high('val_accuracy')

# get the round with the best result
"Rounds to high: ", analyze_object.rounds2high('val_accuracy')

# get the best paramaters
"best_params: ",analyze_object.best_params('val_accuracy', ['accuracy', 'loss', 'val_loss'])

# get correlation for hyperparameters against a metric
"correlate: ", analyze_object.correlate('val_loss', ['accuracy', 'loss', 'val_loss'])

# a regression plot for two dimensions 
"plot_regs: ", analyze_object.plot_regs('val_accuracy', 'val_loss')

# line plot
"plot_line: ", analyze_object.plot_line('val_accuracy')

# up to two dimensional kernel density estimator
"plot_kde: ", analyze_object.plot_kde('val_accuracy')

# a simple histogram
"plot_hist: ", analyze_object.plot_hist('val_accuracy', bins=50)

# heatmap correlation
"plot_corr:", analyze_object.plot_corr('val_loss', ['accuracy', 'loss', 'val_loss'])

# a four dimensional bar grid
"plot_bars:", analyze_object.plot_bars('BATCH_SIZE', 'val_accuracy', 'first_neuron', 'LR')

evaluate_object = talos.Evaluate(scan_object)
print("Evaluate: ", evaluate_object.evaluate(X_train, y_train, folds=10, metric='val_accuracy', task='multi_label'))
# model.summary()


# talos.Deploy(scan_object=scan_object, model_name='iris_deploy', metric='val_acc');


###
# model = model_brain_tumor(X_train, y_train, X_val, y_val, p)[1]
# # predicting the test set results
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5)

# # making the Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)

# print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/51)*100))

# sns.heatmap(cm,annot=True)
# plt.savefig('h.png')
# plt.show()


