# Deep-Doc-Classifier-Using-Bayesian-Neural-Networks
## BayesianNeuralNetworks_for_Deep_Docs:
This file contains the main code for the Bayesian Neural Network for Deep Doc Classification.
The data is divided into training and testing sets, and are preprocessed. During the preprocessing step, single channel  tif image is converted into a three channel image with the first channel repeated twice.  
After the conversion to a 3 channel image, all the images are resized to a 277x227 image, this is done for reducing the processing power required.  
After resizing the images along with their labels are strored in the preprocessed_train.p and preprocessed_test.p files for both the train and test datsets respectively, this ensures that the images can be accessed faster for training any other model.  
  
  
All the handling of data from conversion to resize to creating preprocessed_train.p and preprocessed_test.p is handle by the ### util.py file.  
  
After the data is saved, it is loaded and uitl.build_data_pipeline is called, it returns iterators for train and test along with the images and labels.  
  
Once the images are returned they are passed to ### bayesian_cnn.bayesian_alexnet. This file containes the bayesian alexnet.The model is similar to the actual Alexnet, except for one difference that instead of tensorflow Convolution2D layers tensorflow_probability Convolution2DFlipout layers are used and tensorflow Dense layers tensorflow_probability DenseFlipout layers are used.  
  
Once the model is created, it is used to minimise the ELBO loss. Also posterior statistics weights are extracted for layers with weight distributions for visualization.  And model is trained, and accuracy is ploted along with the Weight Means and Standard deviation.
