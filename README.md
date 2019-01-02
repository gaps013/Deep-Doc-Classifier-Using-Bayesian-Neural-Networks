# Deep-Doc-Classifier-Using-Bayesian-Neural-Networks
## Requirements:
- tensorflow  
- tensorflow probability
- keras  
- pandas  
- matplotlib  
- numpy  
- PIL 
- skimage  
- seaborn  
  
## Files Description:
## BayesianNeuralNetworks_for_Deep_Docs:
This file contains the main code for the Bayesian Neural Network for Deep Doc Classification.
The data is divided into training and testing sets, and are preprocessed. During the preprocessing step, single channel  tif image is converted into a three channel image with the first channel repeated twice.  
After the conversion to a 3 channel image, all the images are resized to a **277x227** image, this is done for reducing the processing power required.  
After resizing the images along with their labels are strored in the **preprocessed_train.p** and **preprocessed_test.p** files for both the train and test datsets respectively, this ensures that the images can be accessed faster for training any other model.  
  
All the handling of data from conversion to resize to creating **preprocessed_train.p** and **preprocessed_test.p** is handle by the **util.py file**.  
  
After the data is saved, it is loaded and uitl.build_data_pipeline is called, it returns iterators for train and test along with the images and labels.  
  
Once the images are returned they are passed to **bayesian_cnn.bayesian_alexnet**. This file containes the bayesian alexnet.The model is similar to the actual Alexnet, except for one difference that instead of tensorflow Convolution2D layers tensorflow_probability **Convolution2DFlipout** layers are used and tensorflow Dense layers tensorflow_probability **DenseFlipout** layers are used.  
  
Once the model is created, it is used to minimise the **ELBO loss**. Also posterior statistics weights are extracted for layers with weight distributions for visualization.  And model is trained, and accuracy is ploted along with the Weight Means and Standard deviation.  
## config.py: 
This file handles the parameters such as:  
- start_epoch = 1                               ->**Initial epoch.**  
- num_classes=10                                  ->**Number of class labels.**  
- learning_rate = 0.0001                          ->**Initial learning rate.**  
- momentum = 0.9                                  ->**Initial Momemntum for SGD.**  
- decay = 0.0005                                  ->**Initial Decay for SGD.**  
- epochs = 10                                     ->**Number of epochs to train for.**  
- batch_size = 1                                  ->**Batch size.**  
- eval_freq = 20                                  ->**Frequency at which to validate the model.**  
- num_monte_carlo = 50                            ->**Network draws to compute predictive probabilities.**  
- architecture = 'alexnet'                        ->**Network architecture to use.**  
- kernel_posterior_scale_mean = -9.0              ->**Initial kernel posterior mean of the scale (log var) for q(w)**  
- kernel_posterior_scale_constraint = 0.2         ->**Posterior kernel constraint for the scale (log var) of q(w).**  
- kernel_posterior_scale_stddev = 0.1  
- kl_annealing = 50                               ->**Epochs to anneal the KL term (anneals from 0 to 1).**  
- subtract_pixel_mean = False                     ->**Boolean for normalizing the images.**  
  
