"""
BRAIN CANCER - IS OR NOT?

Here we can found functions in charge of creating plots

# 1 concerns functions creating plots basic on one iteration
Linear plots.
It could be used for one, the best set of hyperparameters

# 2 concerns functions which allow presenting differences between iterations, where their extension depend on variable "LIMIT" in "ConvNet.py" 
Box plots

"""
# necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 1______________________________________________
 # visualize accuracy and loss 
def singleRound(history):                      
    # four plots together_____________________
    _, axs = plt.subplots(2,2)
    axs[0,0].plot(history.history['loss'], 'r') # train loss
    axs[0,0].set_ylabel("Error")
    axs[0,0].set_xlabel("Epochs")
    axs[0,0].set_title("loss")
    axs[0,0].grid()

    axs[0,1].plot(history.history['accuracy'], 'g')  # train accuracy
    axs[0,1].set_ylabel("Accuracy")
    axs[0,1].set_xlabel("Epochs")
    axs[0,1].set_title("accuracy")
    axs[0,1].grid()

    axs[1,0].plot(history.history['val_loss'], 'r') # val loss
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
    # plt.savefig('val_train_loss.png')
    plt.show()

    # 2 acc
    plt.close()
    plt.plot(history.history['val_accuracy'], 'r', history.history['accuracy'], 'b')
    plt.grid()
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["val_acc", "train_acc"])
    # plt.savefig('val_train_acc.png')
    plt.show()

    return

# 2______________________________________________
# def multipleIterations(data):

#     return
