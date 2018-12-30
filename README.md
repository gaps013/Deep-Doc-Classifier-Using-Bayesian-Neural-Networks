# Deep-Doc-Classifier-Using-Bayesian-Neural-Networks
## BayesianNeuralNetworks_for_Deep_Docs:
This file contains the main code for the Bayesian Neural Network for Deep Doc Classification.
The data is divided into training and testing sets, and are preprocessed. During the preprocessing step, single channel  tif image is converted into a three channel image with the first channel repeated twice.
After the conversion to a 3 channel image, all the images are resized to a 277x227 image, this is done for reducing the processing power required.
After resizing the images along with their labels are strored in the preprocessed_train.p and preprocessed_test.p files for both the train and test datsets respectively, this ensures that the images can be accessed faster for training any other model.
