import os
from PIL import Image
import shutil, random
import numpy as np
from skimage import transform
from skimage import io
from os import walk
import warnings
import pickle
import tensorflow as tf
import config as cf
warnings.filterwarnings("ignore")


def to3_channel(train_dir):
    for folders in os.listdir(train_dir):
        current_dir = os.path.join(train_dir,folders)
        if(os.path.isdir(current_dir)):
            for files in os.listdir(current_dir):
                if(files.endswith('.tif')):
                    img = Image.open(os.path.join(current_dir,files))
                    imarray = np.array(img)
                    stacked_img = np.stack((imarray,)*3, axis=-1)
                    three_channel_image = Image.fromarray(stacked_img,'RGB')
                    three_channel_image.save(os.path.join(current_dir,files), "JPEG")

def get_class(folder_name):
    class_names = ['ADVE','Email','Form','Letter','Memo','News','Note','Report','Resume','Scientific']
    if(folder_name in class_names):
        return class_names.index(folder_name)


def preprocess_and_save_data(data_type):
    root_dir = 'Data'
    if(data_type=='train'):     
        imgs = []
        labels = []

        root_dir_train = os.path.join(root_dir,'Train')
        for folders in os.listdir(root_dir_train):
            for files in os.listdir(os.path.join(root_dir_train,folders)):
                if (files.endswith('.tif')):
                    img = io.imread(os.path.join(root_dir_train,folders,files),plugin='pil')
                    img = transform.resize(img, (227, 227))
                    imgs.append(img)

                    label = get_class(folders)
                    labels.append(label)
                    
        X_train = np.array(imgs, dtype='float32')
        # Make one hot targets
        Y_train = np.array(labels, dtype = 'uint8')

        train_data = {"features": X_train, "labels": Y_train}
        if not os.path.exists(os.path.join(root_dir,"Preprocessed_Data")):
                os.makedirs(os.path.join(root_dir,"Preprocessed_Data"))
        pickle.dump(train_data,open(os.path.join(root_dir,"Preprocessed_Data","preprocessed_train.p"),"wb"))

        return train_data

    elif(data_type=='test'):
        imgs = []
        labels = []

        root_dir_test = os.path.join(root_dir,'Test')
        for folders in os.listdir(root_dir_test):
            for files in os.listdir(os.path.join(root_dir_test,folders)):
                if (files.endswith('.tif')):
                    img = io.imread(os.path.join(root_dir_test,folders,files),plugin='pil')
                    img = transform.resize(img, (227, 227))
                    imgs.append(img)

                    label = get_class(folders)
                    labels.append(label)
                    
        X_test = np.array(imgs, dtype='float32')
        # Make one hot targets
        Y_test = np.array(labels, dtype = 'uint8')

        test_data = {"features": X_test, "labels": Y_test}
        if not os.path.exists(os.path.join(root_dir,"Preprocessed_Data")):
            os.makedirs(os.path.join(root_dir,"Preprocessed_Data"))
        pickle.dump(test_data,open(os.path.join(root_dir,"Preprocessed_Data","preprocessed_test.p"),"wb"))
        
        return test_data

def build_data_pipeline(X_train, X_test,y_train, y_test):
    '''
    Dataset iterator for training the model
    :param X_train: Numpy array consisting of train images
    :param X_test: Numpy array consisting of test images
    :param y_train: Numpy array containing train labels
    :param y_test: Numpy arrray containing test labels
    :return: iterators for train and test
    '''

    train_data = tf.data.Dataset.from_tensor_slices((np.float32(X_train), np.int32(y_train)))
    train_batches = train_data.shuffle(50000, reshuffle_each_iteration=True).repeat().batch(cf.batch_size)
    train_iterator = train_batches.make_one_shot_iterator()

    # Building an iterator with test_dataset with batch_size = X_test.shape[0]. We use entire testing data for one shot iterator
    test_data = tf.data.Dataset.from_tensor_slices((np.float32(X_test),np.int32(y_test)))
    test_frozen = (test_data.take(X_test.shape[0]).repeat().batch(X_test.shape[0]))
    test_iterator = test_frozen.make_one_shot_iterator()

    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    iter_handle = tf.placeholder(tf.string, shape=[])
    iterator_feed = tf.data.Iterator.from_string_handle(iter_handle, train_batches.output_types, train_batches.output_shapes)
    images, labels = iterator_feed.get_next()

    return images, labels, iter_handle, train_iterator, test_iterator