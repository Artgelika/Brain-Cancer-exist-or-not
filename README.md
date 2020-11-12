That will be a neural network that predicts the presence of cancer in the brain. After that, it could be extended to do something else.
https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection?fbclid=IwAR1R8lig76z4E8mGS0mz9727U5PTatLmcZjJQ2tk0GlVkKtW93obJ_H6V0E

Structure of my network:
1. Prepare photos' names in file "Zmiana_nazwy.py".
2. Using that photos in "CNN.py", changing them into numbers and saving.
3. Using data prepared before (in CNN.py). Creating a model with a train set and fitting with additional validation set in "ConvNet.py".
