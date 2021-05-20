#!/usr/bin/env python
# coding: utf-8

# # Import Libraries
# 

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.decomposition import PCA

from tensorflow import keras

from tensorflow.keras.datasets import cifar10

import tensorflow as tf
import argparse

import matplotlib.pyplot as plt
import numpy as np

import pickle
import os
import csv

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.summary import summary

from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


from tripletLossHelper import pairwise_distance, masked_maximum, masked_minimum
from evaluateEmbeddings import clus_acc,test2NN,nmi,get_embeddings_labels



try:
    # pylint: disable=g-import-not-at-top
    from sklearn import metrics

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# In[3]:


print (tf.__version__)
print (tf.test.is_gpu_available())


# # Parse the Hyperparameters

# In[4]:


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-size',
                        type=int,
                        default=32,
                        help='Embedding size for the model')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1024,
                        help='Batch size for training')
    parser.add_argument('--epochs',
                        type=int,
                        default=500,
                        help='epochs to train')
    parser.add_argument('--sigma',
                        type=int,
                        default=64,
                        help='sigma to select triplets')
    parser.add_argument('--flip',
                        type=int,
                        default=1,
                        help='use 1/sigma instead')
    parser.add_argument('--lambdas',
                        type=int,
                        default=0,
                        help='for reconstruction loss')
    parser.add_argument('--save-path',
                       type=str,
                        help='path where model weights should be saved',
                       default='triplet_encoder-cv')
    parser.add_argument('--folds',
                        type=int,
                        default=5,
                        help='number of folds to select the hyperparameters')
    args = parser.parse_known_args()[0]
    return args


# In[5]:


args = parse_arguments()


# In[6]:


print(args)


# In[7]:


args.lambdas = 0.1*args.lambdas


# # Define Loss Function
# 

# ## Define our Triplet Loss

# In[8]:


step_hist= 1


# In[9]:


def triplet_loss_cus(anchor, positive, negative):
    """Calculate the loss

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive weighted images.
      negative: the embeddings for the negative weighted images.

    Returns:
      the loss according to MsDNN as a float tensor.
    """
    # with tf.compat.v1.variable_scope('triplet_loss'):

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)

    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.subtract(pos_dist, neg_dist)

    loss = tf.reduce_mean(tf.math.log_sigmoid(-1 * basic_loss), 0)
    # print (loss)
    return -1 * loss


# In[10]:


def triplet_semihard_without_grad(embeddings_labels, sigma1=1.0, sigma2=1.0):
    '''
    This function is used to calculate the probabilities for positive and negative weighted embeddings
    Input:
    embeddings: the embeddings which represents the images in other space where distance is interpreable
    labels: corresponding label for each embeddings
    sigma: tells how much weight to be assigned based on distance. Low sigma will assign high weight to
            nearby sample.
    
    Output: positive and negative probabilties
    '''
    
    # First we extract the labels and embeddings
    labels = embeddings_labels[:, :1]

    labels = tf.cast(labels, dtype='int32')

    embeddings = embeddings_labels[:, 1:]
    

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)

    
    # This matrix will have 1 when labels are same and 0 when they are different
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))                               
    # Invert so we can select negatives only.
    # This matrix will have 1 when labels are different and 0 when they are same                           
    adjacency_not = math_ops.logical_not(adjacency)
    
    #Infer batch size
    batch_size = array_ops.size(labels)

    # For calculating positive probability
    affinity = math_ops.exp(-pdist_matrix / sigma1) - array_ops.diag(array_ops.ones([batch_size]))
    d_a_p = math_ops.multiply(math_ops.cast(
        adjacency, dtype=dtypes.float32), affinity)
    pos_prob = math_ops.divide(d_a_p, tf.reduce_sum(d_a_p, axis=1, keepdims=True))

    # Set pos-prob of nearest to 1 in case of nan.
    mask_is_nan = tf.tile(tf.math.is_nan(tf.reduce_sum(pos_prob, axis=1, keepdims=True)),
                          [1, embeddings.shape.as_list()[0]])
    
    pdist_matrix_pos = math_ops.multiply(math_ops.cast(
        adjacency, dtype=dtypes.float32), pdist_matrix)
    
    select_nearest = tf.cast(
                tf.math.equal
                (pdist_matrix_pos,tf.reduce_max(
                    pdist_matrix_pos,
                    axis=1,
                    keepdims=True
                    )
                ),
                dtype=dtypes.float32
            )
    
    pos_prob = array_ops.where(mask_is_nan, select_nearest, pos_prob)

    # For calculating negative probability
    affinity = math_ops.exp(-pdist_matrix / sigma2) - array_ops.diag(array_ops.ones([batch_size]))
    d_a_n = math_ops.multiply(math_ops.cast(
        adjacency_not, dtype=dtypes.float32), affinity)
    neg_prob = math_ops.divide(d_a_n, tf.reduce_sum(d_a_n, axis=1, keepdims=True))

    # Set neg-prob of nearest to 1 in case of nan.
    mask_is_nan = tf.tile(tf.math.is_nan(tf.reduce_sum(neg_prob, axis=1, keepdims=True)),
                          [1, embeddings.shape.as_list()[0]])
    pdist_matrix_neg = math_ops.multiply(math_ops.cast(
        adjacency, dtype=dtypes.float32), pdist_matrix)
    
    select_nearest = tf.cast(
                tf.math.equal
                (pdist_matrix_pos,tf.reduce_max(
                    pdist_matrix_neg,
                    axis=1,
                    keepdims=True
                    )
                ),
                dtype=dtypes.float32
            )
#     print (select_nearest)    
    neg_prob = array_ops.where(mask_is_nan, select_nearest, neg_prob)

    return pos_prob, neg_prob


# In[11]:


def custom_loss(embeddings, pos_prob, neg_prob, verbose=False):
    positives = tf.matmul(pos_prob, embeddings)
    negatives = tf.matmul(neg_prob, embeddings)

    return triplet_loss_cus(embeddings, positives, negatives)


# # Define our base_model

# In[12]:


def encoder(image_input_shape, embedding_size):
    """
    Base network to be shared (eq. to feature extraction).
    """
    input_image = tf.keras.Input(shape=image_input_shape)

    x = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3),strides=(1,1), activation='relu', padding='SAME')(input_image)
#     x = tf.keras.layers.AveragePooling2D( padding='SAME')(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),strides=(2,2), activation='relu', padding='SAME')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),strides=(1,1),activation='relu', padding='SAME')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),strides=(2,2),activation='relu', padding='SAME')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),strides=(2,2),activation='relu', padding='SAME')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),strides=(1,1),activation='relu', padding='SAME')(x)

#     x = tf.keras.layers.AveragePooling2D(padding='SAME')(x)
    x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(128,activation='relu')(x)
    x = tf.keras.layers.Dense(embedding_size,activation=None)(x)

    base_network = tf.keras.Model(inputs=input_image, outputs=x,name='Encoder')
    tf.keras.utils.plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
    return base_network


# In[13]:


def decoder(image_input_shape, embedding_size):
    input_embeddings = tf.keras.layers.Input(shape=embedding_size, name='input_embedding')
#     x = tf.keras.layers.Dense(128, activation='relu')(input_embeddings)
    x = tf.keras.layers.Dense(512, activation='relu')(input_embeddings)
    x = tf.keras.layers.Reshape(target_shape=(4, 4, 32))(x)
#     x = tf.keras.layers.UpSampling2D((2, 2))(x)
#     x = tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=(3,3),activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=(3,3),strides=(2,2),padding="SAME",activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=(3,3),strides=(2,2),padding="SAME",activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=(1,1),padding="SAME",activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=(3,3),strides=(1,1),padding="SAME",activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=2,kernel_size=(3,3),strides=(1,1),padding="SAME",activation='relu')(x)

    #     x = tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=(3,3),activation='relu')(x)
#     x = tf.keras.layers.UpSampling2D((2, 2))(x)
#     x = tf.keras.layers.Conv2DTranspose(filters=6,kernel_size=(3,3),activation='relu',strides=(2,2))(x)
#     x = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=(3,3),padding="SAME",activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")(x)
    
    decoder_network = tf.keras.Model(inputs=input_embeddings, outputs=x)
    tf.keras.utils.plot_model(decoder_network, to_file='decoder_network.png', show_shapes=True, show_layer_names=True)
    return decoder_network


# In[14]:


def clus(embedding_size):
    input_embeddings = tf.keras.layers.Input(shape=embedding_size, name='input_embedding')
    x = tf.math.l2_normalize(input_embeddings, axis=1)
    clus_network = tf.keras.Model(inputs=input_embeddings, outputs=x)
    tf.keras.utils.plot_model(clus_network, to_file='clus_network.png', show_shapes=True, show_layer_names=True)
    return clus_network


# In[15]:


input_image_shape = (16, 16, 1)


# In[16]:


enc = encoder(input_image_shape,args.embedding_size)


# In[17]:


dec = decoder(input_image_shape, args.embedding_size)


# In[18]:


clus = clus(args.embedding_size)


# In[19]:


tf.keras.utils.plot_model(clus, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[20]:


tf.keras.utils.plot_model(enc, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[21]:


tf.keras.utils.plot_model(dec, to_file='model.png', show_shapes=True, show_layer_names=True)


# # Loading data

# In[22]:


import h5py 
from functools import reduce
def hdf5(path, data_key = "data", target_key = "target", flatten = True):
    """
        loads data from hdf5: 
        - hdf5 should have 'train' and 'test' groups 
        - each group should have 'data' and 'target' dataset or spcify the key
        - flatten means to flatten images N * (C * H * W) as N * D array
    """
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get(data_key)[:]
        y_tr = train.get(target_key)[:]
        test = hf.get('test')
        X_te = test.get(data_key)[:]
        y_te = test.get(target_key)[:]
        if flatten:
            X_tr = X_tr.reshape(X_tr.shape[0], reduce(lambda a, b: a * b, X_tr.shape[1:]))
            X_te = X_te.reshape(X_te.shape[0], reduce(lambda a, b: a * b, X_te.shape[1:]))
    return X_tr, y_tr, X_te, y_te


# In[23]:


x_train, y_train, x_test, y_test = hdf5("usps.h5")
x_train.shape, x_test.shape


# In[24]:


# num_samples = 10
# num_classes = len(set(y_train))

# classes = set(y_train)
# num_classes = len(classes)
# fig, ax = plt.subplots(num_samples, num_classes, sharex = True, sharey = True, figsize=(num_classes, num_samples))

# for label in range(num_classes):
#     class_idxs = np.where(y_train == label)
#     for i, idx in enumerate(np.random.randint(0, class_idxs[0].shape[0], num_samples)):
#         ax[i, label].imshow(x_train[class_idxs[0][idx]].reshape([16, 16]), 'gray')
#         ax[i, label].set_axis_off()


# ## Functions for loading the training/validation/testing DATA, as well as some other parameters

# In[25]:


def correct_shape(x, y, size):
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.reshape(x, [size, 16, 16, 1])
    y = tf.reshape(y, [size, 1])
    return x, y
def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def scale(x, min_val=0.0, max_val=255.0):
    x = tf.cast(x, dtype=tf.float32)
    return math_ops.div(math_ops.subtract(x, min_val), math_ops.subtract(max_val, min_val))


# ## Load data

# In[26]:


x_train,y_train = correct_shape(x_train,y_train,7291)


# In[27]:


x_test,y_test = correct_shape(x_test,y_test,2007)


# In[28]:


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(lambda x, y: (scale(x), y)).shuffle(args.batch_size).batch(args.batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(lambda x, y: (scale(x), y)).shuffle(args.batch_size).batch(args.batch_size)


# In[29]:


# # Define the K-fold Cross Validator
# kfold = KFold(n_splits=args.folds)


# In[30]:


# train_ds_list = []
# val_ds_list = []
# for train_index,test_index in kfold.split(x_train):
    
#     x_train_cv = x_train[train_index]
#     y_train_cv = y_train[train_index]
    
#     x_val_cv = x_train[test_index]
#     y_val_cv = y_train[test_index]
    
#     x_train_cv,y_train_cv = correct_shape(x_train_cv,y_train_cv,x_train_cv.shape[0])
#     x_val_cv,y_val_cv = correct_shape(x_val_cv,y_val_cv,x_val_cv.shape[0])
    
#     train_ds = tf.data.Dataset.from_tensor_slices((x_train_cv, y_train_cv))
#     train_ds = train_ds.map(lambda x, y: ((x), y)).shuffle(args.batch_size).batch(args.batch_size)
    
#     train_ds_list.append(train_ds)
    
#     val_ds = tf.data.Dataset.from_tensor_slices((x_val_cv, y_val_cv))
#     val_ds = val_ds.map(lambda x, y: ((x), y)).shuffle(args.batch_size).batch(args.batch_size)
    
#     val_ds_list.append(val_ds)


# In[31]:


# train_ds_list


# In[32]:


# val_ds_list


# In[33]:


input_image_shape = (16, 16, 1)


# ## Instantiate the base network and define our model
# 

# In[34]:


# base_network = create_base_network(input_image_shape, args.embedding_size)

input_images = tf.keras.layers.Input(shape=input_image_shape, name='input_image') # input layer for images
input_labels = tf.keras.layers.Input(shape=(1,), name='input_label')    # input layer for labels
# input_embeddings = tf.keras.layers.Input(shape=embe+dding_size, name='input_embedding')
embeddings = enc([input_images])               # output of network -> embeddings

norm_embeddings = clus([embeddings])

labels_plus_embeddings = tf.keras.layers.concatenate([input_labels, norm_embeddings])  # concatenating the labels + embeddings

reconstructed_image = dec([embeddings])

# Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
model = tf.keras.Model(inputs=[input_images, input_labels],
              outputs=[labels_plus_embeddings,reconstructed_image])

model.summary()
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# In[35]:


optimizer = tf.optimizers.Adam(learning_rate=0.00001)


# In[36]:


print (type(train_ds))


# # Training our model

# In[37]:


if args.flip:
    args.sigma = 1/args.sigma
sigma1 = args.sigma
sigma2 = args.sigma


# In[38]:


import os
if not os.path.exists(str(args.save_path)):
    os.makedirs(str(args.save_path))


# In[39]:


import os
if not os.path.exists(args.save_path + "/"+str(args.lambdas)+"/model_"+str(args.sigma)):
    os.makedirs(args.save_path + "/"+str(args.lambdas)+"/model_"+str(args.sigma))


# In[40]:


# model.save(before_training_model_path)
# # model.save_weights('before_training_model.h5')
# for fold,train_ds in enumerate(train_ds_list):
#     print ("Fold:",fold)
#     model = tf.keras.models.load_model(before_training_model_path)
    for i in range(args.epochs):
        epochloss = 0
        den = 0
        for (batch, (images, labels)) in enumerate(train_ds):
            den += 1

            # For learning changing the labels and creating super classes
            labels = [1 if y > 4 else 0 for y in tf.reshape(labels, [-1])]
            images, labels = correct_shape(images, labels, images.shape[0])
            labels = tf.dtypes.cast(labels, tf.float32) 
            labels = tf.reshape(labels,[-1,1])

            # Calculate the probabilities for positive and negative weighted embeddings
            embeddings_labels, reconstructed_image = model([images, labels])
            pos_prob, neg_prob = triplet_semihard_without_grad(
            embeddings_labels, sigma1, sigma2)

            # Train
            with tf.GradientTape() as tape:
                embeddings_labels, reconstructed_image = model([images, labels])
                current_loss = custom_loss(embeddings_labels[:, 1:], pos_prob, neg_prob) + tf.multiply(tf.convert_to_tensor(args.lambdas, dtype=tf.float32),tf.losses.mean_squared_error(tf.reshape(images,[-1]),tf.reshape(reconstructed_image,[-1])))
                epochloss += current_loss

            # Update Gradients
            gradients = tape.gradient(current_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


        print("Loss after epoch")
        print(i)
        print(epochloss/den)

#         if i%10 ==0:
#     print("Evaluating and saving weights")
#     import os
#     if not os.path.exists(args.save_path + "/"+str(args.lambdas)+"/model_"+str(args.sigma)):
#         os.makedirs(args.save_path + "/"+str(args.lambdas)+"/model_"+str(args.sigma))
#     model.save(args.save_path + "/"+str(args.lambdas)+"/model_"+str(args.sigma)+"/triplet_"+str(i)+'.h5')

#     # Clustering acc and NMI: for 10 classes (subclasses)
#     embeddings_labels = get_embeddings_labels(model,train_ds)
#     embeddings = embeddings_labels[:, 1:]
#     labels = embeddings_labels[:, :1]
#     labels = tf.cast(labels, dtype='int32')
#     kmeans = KMeans(init='k-means++', n_clusters=10, n_init=100)
#     kmeans.fit(embeddings)
#     print("NMI,Clustering Accuracy: "+ str((nmi( kmeans.labels_,np.asarray(tf.reshape(labels,[-1]))),clus_acc( kmeans.labels_,np.asarray(tf.reshape(labels,[-1]))))))

# #     KNN for superclasses
#     print("KNN: " + str(test2NN(model,train_ds,test_ds)))
# #     ktest = test2NN(model,train_ds,val_ds_list[fold])
# #     print(ktest)


# In[41]:


print("Evaluating and saving weights")
import os
if not os.path.exists(args.save_path + "/"+str(args.lambdas)+"/model_"+str(args.sigma)):
    os.makedirs(args.save_path + "/"+str(args.lambdas)+"/model_"+str(args.sigma))
model.save(args.save_path + "/"+str(args.lambdas)+"/model_"+str(args.sigma)+"/triplet_"+str(args.epochs)+'.h5')

# Clustering acc and NMI: for 10 classes (subclasses)
embeddings_labels = get_embeddings_labels(model,train_ds)
embeddings = embeddings_labels[:, 1:]
labels = embeddings_labels[:, :1]
labels = tf.cast(labels, dtype='int32')
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=100)
kmeans.fit(embeddings)
print("NMI,Clustering Accuracy: "+ str((nmi( kmeans.labels_,np.asarray(tf.reshape(labels,[-1]))),clus_acc( kmeans.labels_,np.asarray(tf.reshape(labels,[-1]))))))

# KNN for superclasses
print("KNN: " + str(test2NN(model,train_ds,test_ds)))


# In[ ]:




