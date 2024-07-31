from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import os
from datetime import datetime
from config import Config, ConfigDTS
import detector_new.dataset as dataset
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import scipy as sp
import detector_new.draw_image as draw_image
#from keras_grad_cam.pyimagesearch.gradcam import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from sklearn.model_selection import train_test_split
import shutil
import pickle_load_save
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
import itertools
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
# actfunc = 'relu'
actfunc = tf.keras.layers.LeakyReLU(alpha=0.1)
import random
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow import keras
import math
import os.path as path
import logging
import time
import gc
from numpy.random import seed
from tensorflow.compat.v1.random import set_random_seed
import keras_tuner as kt


_input_shape = 0

def make_model_with_global_average_pooling(input_shape=(13, 13, 512)):
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(512, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dense(60, activation='relu')(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_model_with_global_average_pooling_two_op(input_shape=(13, 13, 512)):
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(512, (3, 3), padding="same",
                            activation='relu')(inputs)
    # x = keras.layers.Conv2D(512, (3, 3), padding="same",
    #                         activation='relu')(inputs)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


#Val_ACC: 88.93 Train_ACC: 99
def make_model(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')

    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


#Val_ACC: 88.93 Train_ACC: 100
def make_model_all_cnn(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(512, (3, 3), padding="same",
                            activation='relu')(inputs)

    # x = keras.layers.BatchNormalization()(x)

    # x = keras.layers.MaxPool2D((2, 2), padding="same")(x)
    # x = keras.layers.Dropout(0.25)(x)
    # x = keras.layers.Conv2D(128, (3, 3), padding="same",
    #                         activation='relu')(inputs)
    # x = keras.layers.MaxPool2D((2, 2), padding="same")(x)
    # x = keras.layers.Conv2D(256, (3, 3), padding="same",
    #                         activation='relu')(inputs)
    # x = keras.layers.MaxPool2D((2, 2), padding="same")(x)
    # x = keras.layers.Conv2D(512, (3, 3),
    #                         activation='relu')(inputs)
    x = keras.layers.Conv2D(256, (1, 1), activation='relu')(x)
    # x = keras.layers.Conv2D(10, (1, 1), activation='relu')(x)
    x = keras.layers.Flatten()(x)

    # model.add(Conv2D(64, (1, 1), activation='relu'))

    # x = keras.layers.Dropout(0.25)(x)
    # x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(64)(x)
    # x = keras.layers.Dropout(0.25)(x)
    # x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(64, kernel_initializer='uniform')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10, kernel_initializer='uniform')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10, kernel_initializer='uniform')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=inputs, outputs=x)

    return model


def make_3d_model(input_shape=(13, 13, 6)) -> keras.models.Model:
    input_shape = (10, 13, 13, 512)
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv3D(64, 3, padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool3D((2, 2, 2), padding="same")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv3D(128, 3, padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool3D((3, 3, 3), padding="same")(x)
    x = keras.layers.Conv3D(256, 3, padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool3D((3, 3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    return model

#Val_ACC: 85.74 Train_ACC: 100
def make_model_bn(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu')(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    return model

	
def make_simple_model_two_neurons(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_large_model_two_neurons(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')

    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_large_model_two_neurons_new(input_shape=(13, 13, 30)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')

    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model	
	

def make_large_model_two_neurons_new_nn(input_shape=(13, 13, 1)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')

    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model

    inputs = keras.layers.Input(shape=input_shape, name='main_input')

    # Add more regularization
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.LeakyReLU()(x)

    # Reduce the complexity of the model
    x = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.LeakyReLU()(x)

    # # Use data augmentation
    # x = keras.layers.experimental.preprocessing.RandomRotation(0.1)(x)
    # x = keras.layers.experimental.preprocessing.RandomZoom(0.1)(x)
    # x = keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal')(x)

    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_model_five_labels(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.BatchNormalization()(inputs)
    # y = keras.layers.Conv2D(12, (5, 5), padding="same", activation='relu')(x)
    # y = keras.layers.Conv2D(24, (5, 5), activation='relu')(y)
    # y = keras.layers.MaxPool2D((2, 2), padding="same")(y)
    # y = keras.layers.Flatten()(y)

    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')(x)
    x = keras.layers.MaxPool2D((2, 2))(x)
    # x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D((2, 2), padding="same")(x)
    x = keras.layers.Flatten()(x)

    # x = keras.layers.Dense(600)(x)
    # x = keras.layers.LeakyReLU()(x)

    # m = keras.layers.Add()([x, y])

    x = keras.layers.Dense(100)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dropout(0.50)(x)
    x = keras.layers.Dense(80)(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Dense(5, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_simple_model_old(input_shape=(13, 13, 50)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    # x = keras.layers.Conv2D(512, (3, 3), padding="same", activation='relu')(inputs)
    # #x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # x = keras.layers.Dropout(0.15)(x)
    # x = keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    # #x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')(inputs)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.35)(x)
    # x = keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    # #x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # x = keras.layers.Dropout(0.45)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_simple_model_VGG_s(input_shape=(13, 13, 50)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')(inputs)
    #x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # x = keras.layers.Dropout(0.15)(x)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.25)(x)
    # x = keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu')(inputs)
    # #x = keras.layers.BatchNormalization()(x)
    # # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # # x = keras.layers.Dropout(0.15)(x)
    # x = keras.layers.Conv2D(16, (3, 3), padding="same", activation='relu')(x)
    # #x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # x = keras.layers.Dropout(0.25)(x)
    # x = keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    # #x = keras.layers.BatchNormalization()(x)
    # # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # # x = keras.layers.Dropout(0.35)(x)
    # x = keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    # #x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # x = keras.layers.Dropout(0.45)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_simple_model_multiclass(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(512, (3, 3), padding="same", activation='relu')(inputs)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.15)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.35)(x)
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.45)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(2, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    return model


def make_simple_model_rfe(input_shape=(13, 13, 512)):
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#Paper 1
def make_simple_model_paper(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    return model

#Paper 2
def make_network1_alt(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_ff_model(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_ff_model_2(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


#VGG_Blocks
def model_vgg(input_shape=(13, 13, 512)) -> keras.models.Model:
    filters = max(2**round(math.log2(input_shape[-1])-1), 256)
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(filters, (3, 3), padding="same", activation='relu')(inputs)
    x = keras.layers.Conv2D(filters, (3, 3), padding="same", activation='relu')(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(filters>>1, (3, 3), padding="same", activation='relu')(x)
    x = keras.layers.Conv2D(filters>>1, (3, 3), padding="same", activation='relu')(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(filters>>1, activation='relu')(x)
    # # x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    # # x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


#VGG_Blocks
def model_vgg_(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu')(inputs)
    x = keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.45)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def make_simple_model_seq(input_shape=(13, 13, 512)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu', name="firstlayer", input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((3, 3), padding="same"))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((3, 3), padding="same"))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128))
    model.add(keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(16))
    model.add(keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model


def BuildHyperVGGModel(hp):
    global _input_shape
    filters = max(2**round(math.log2(_input_shape[-1])-1), 256)
    # # filters = [int(filters*(i+1)/4) for i in range(4)]
    # # conv1_units = hp.Choice('conv1_units',values=filters)
    # # conv2_units = hp.Choice('conv2_units',values=filters)
    # # dense_units = hp.Choice('dense_units',values=filters)
    drops = [i*0.05 for i in range(11)]
    inputs = keras.layers.Input(shape=_input_shape, name='main_input')
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation='relu')(inputs)
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(rate=hp.Choice('drop1',values=drops))(x)
    x = keras.layers.Conv2D(filters>>1, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = keras.layers.Conv2D(filters>>1, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(rate=hp.Choice('drop2',values=drops))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(filters>>1)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(rate=hp.Choice('drop3',values=drops))(x)
    x = keras.layers.Dense(16)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)

    # # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = compile_model(model, loss='binary_crossentropy')

    return model


def compile_model(model, lr=0.001, optimizer='adam', loss='mean_squared_error') -> keras.models.Model:
    opt = keras.optimizers.Adam(lr=lr)

    model.compile(optimizer=opt, loss=loss,
                  metrics=['accuracy'])
    return model


def train_old():
    class_name = 'person'
    num = '1'
    input_data=dataset.get_detector_dataset(
        'val', Config.class_names)
    x_train, y_train, meta_train = input_data

    x_val, y_val, meta_val = input_data
    x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=25)
    
	
    # x_combined = np.concatenate((x_train, x_val))
    # y_combined = np.concatenate((y_train, y_val))
    # meta_combined = np.concatenate((meta_train, meta_val))

    # x_train, x_val, y_train, y_val = train_test_split(x_combined, y_combined,
    #                                                   stratify=y_combined,
    #                                                   test_size=0.2)

    # print(len(X_train), len(X_val))

    # exit(0)
    y_train, y_val = train_test_split(y_train, test_size=0.15, random_state=25)
    

    # x_train = x_train[:, :, :, 0:3]
    # x_val = x_val[:, :, :, 0:3]
    
    print(x_train.shape)
    print(x_val.shape)

    # x_train = np.reshape(x_train, (-1, 13, 13, 3, 2))
    # x_val = np.reshape(x_val, (-1, 13, 13, 3, 2))

    model_name = 'cnn_bn'

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/new_dt_'+model_name+'_'+class_name + num+'{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=3, write_grads=True)
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1, min_lr=0.0001, patience=10, factor=0.4)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, verbose=1)

    callbacks_list = [learning_rate_dec,
                      tensorboard, early_stopping]

    model = make_network1_alt(input_shape=(13, 13, 500))
    #model=make_model_bn()
    # model = make_model(input_shape=(13, 13, 6))
    # model = keras.models.load_model('./checkpoints/person_single_output_2.h5')

    # model = keras.models.load_model('./checkpoints/ndt_person_ff.h5')
    model = compile_model(model, loss='binary_crossentropy')

    model.summary()
    op_dir = './data/model_results/cnn_net1'
    with open(os.path.join(op_dir, 'model_6.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    

    model.fit(x_train, y_train, epochs=50, batch_size=128,
              validation_data=(x_val, y_val), callbacks=callbacks_list)

    model.save('./checkpoints/ndt_person_'+model_name+'.h5')

    test_trained_model_nn(model, model_name, x_val, y_val2=y_val)


def train_n():
    class_name = 'person'
    num = '1'
    input_data=dataset.get_detector_dataset(
        'val', Config.class_names)
    x_train, y_train, meta_train = input_data

    x_val, y_val, meta_val = input_data
    x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, y_val = train_test_split(y_train, test_size=0.15, random_state=25)

    print(x_train.shape)
    print(x_val.shape)

    model_name = 'cnn_bn'

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/new_dt_'+model_name+'_'+class_name + num+'{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=3, write_grads=True)
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1, min_lr=0.0001, patience=10, factor=0.4)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, verbose=1)

    checkpoint_filepath = './checkpoints/ndt_person_'+model_name+'.h5'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    callbacks_list = [learning_rate_dec,
                      tensorboard, early_stopping, model_checkpoint]

    model = make_simple_model()
    model = compile_model(model, loss='binary_crossentropy')
    model.summary()

    op_dir = './data/model_results/cnn_net1'
    with open(os.path.join(op_dir, 'model_6.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    model.fit(x_train, y_train, epochs=50, batch_size=128,
              validation_data=(x_val, y_val), callbacks=callbacks_list)


    test_trained_model_nn(model, model_name, x_val, y_val2=y_val)


def train_multiclass():
    class_name = ['person' , 'car']
    num = '1'
    input_data = dataset.get_detector_dataset('val', Config.class_names)
    x_train, y_train, meta_train = input_data

    x_val, y_val, meta_val = input_data
    x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, y_val = train_test_split(y_train, test_size=0.15, random_state=25)

    print(x_train.shape)
    print(x_val.shape)

    model_name = 'cnn_multiclass'

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/new_dt_'+model_name+'_'+class_name+num +
        '{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=3, write_grads=True)

    # reduce learning rate when the model stops improving
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1, min_lr=0.0001, patience=10, factor=0.4)

    # stop training early when the model stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, verbose=1)

    # save the best model based on validation accuracy
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        './checkpoints/ndt_person_'+model_name+'.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)

    callbacks_list = [learning_rate_dec, tensorboard,
                      early_stopping, model_checkpoint]

    #model = make_network1_alt(input_shape=(13, 13, 512))
    #model=make_model_bn()
    model = make_simple_model_multiclass()
    #model=make_ff_model()
    # model = make_model(input_shape=(13, 13, 6))
    # model = keras.models.load_model('./checkpoints/person_single_output_2.h5')

    # model = keras.models.load_model('./checkpoints/ndt_person_ff.h5')
    model = compile_model(model, loss='binary_crossentropy')

    model.summary()
    op_dir = './data/model_results/cnn_net1'
    with open(os.path.join(op_dir, 'model_6.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    model.fit(x_train, y_train, epochs=50, batch_size=128,
              validation_data=(x_val, y_val), callbacks=callbacks_list)

    # The best model has already been saved by ModelCheckpoint, so we don't need to save the model again here

    model=load_model_from_file(checkpoint_name='./checkpoints/ndt_person_cnn_multi.h5')
    test_trained_model_nn(model, model_name, x_val, y_val2=y_val)	


def HyperParameterTune(train_data=None, project_name='model_vgg'):
  # # tuner = kt.RandomSearch(model_new, objective='val_loss', max_trials=5)
  tuner = kt.Hyperband(
              BuildHyperVGGModel,
              objective='val_accuracy',
              max_epochs=15,
              hyperband_iterations=4,
              factor=3,
              directory='./tuner',
              project_name=project_name,
              overwrite=True
            )

  tuner.search(              
      train_data[0], train_data[1],
	  epochs=15,
      batch_size=50,
      validation_split=0.1,
      verbose=1
    )
  # Get the optimal hyperparameters
  # # best_model = tuner.get_best_models(1)[0]
  # # tuner.search_space_summary()
  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

  # # print("""\nThe hyperparameter search is complete.""")
  # # print(f"""The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.""")
  # # print(f"""The optimal hidden layer activation is {best_hps.get('hidden_activation')}.""")
  # # print(f"""The optimal hidden layers is {best_hps.get('layers')}.""")
  # # for i in range(best_hps.get('layers')):
    # # print(f"""\tHidden layer {i}: {best_hps.get(f'fd{i}_units')}.""")

  return tuner.hypermodel.build(best_hps)
	

def train_bn(selection, class_name, method, step, K, n_features, rfe_time):
    global _input_shape

    seed(25)
    set_random_seed(25)

    start = time.time()
    num = '1'
    meta = []
	    
    input_data = dataset.get_detector_dataset('val', Config.class_names)
    x_train, y_train, meta_train = input_data
    y_train = 1-y_train
    n_samples = len(x_train)
    if selection:
      x_train = x_train[:,:,:, selection]
   
    x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, y_val = train_test_split(y_train, test_size=0.15, random_state=25)
    meta_train, meta_val = train_test_split(meta_train, test_size=0.15, random_state=25)
    print(x_val.dtype)

    # feat = len(x_train[0][0][0])
    # output_file_name = path.join(Config.cs_output_root, str(feat)+'_data.dat')
    # pickle_load_save.save(output_file_name, (np.array(x),
                                             # y, np.array(meta)))
    # logging.info('Dataset saved to {}'.format(output_file_name))
    
    model_name = 'cnn_bn'

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/new_dt_'+model_name+'_'+class_name+num +
        '{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=3, write_grads=True)

    # reduce learning rate when the model stops improving
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1, min_lr=0.00001, patience=10, factor=0.25)
    # # learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.4, min_lr=1e-5, min_delta=0.01, verbose=1)

    # stop training early when the model stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, verbose=1)

    model_file = './checkpoints/ndt_'+class_name+'_'+method.lower()+'_'+model_name+'.h5'
    # save the best model based on validation accuracy
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        model_file,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)

    callbacks_list = [learning_rate_dec, tensorboard,
                      early_stopping, model_checkpoint]

    _input_shape = x_train[0].shape
    model = model_vgg(input_shape = _input_shape)

    # # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = compile_model(model, loss='binary_crossentropy')
    # # model = HyperParameterTune(train_data=(x_train, y_train), project_name=class_name+'_'+method.lower()+'_'+model_name)

    model.summary()
    model.fit(x_train, y_train, epochs=50, batch_size=64,
              validation_data=(x_val, y_val), callbacks=callbacks_list)

    end = time.time()
    train_time = end - start	

    """" output directory name """
    op_dir = './data/model_results/'+class_name+'/'+method
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    op_name = 'Step:'+str(step)+'_'+'K_Fold:'+ str(K)+'_'+'Samples:'+str(n_samples)+'_'+ str(current_time)
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)
    op_dir = os.path.join(op_dir, op_name)
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    """" test the model """
    model = load_model_from_file(checkpoint_name=model_file)
    accuracy, TPR, FPR, th, test_time = test_trained_model_nn(model, class_name, method, op_dir, x_val, y_val, meta_val)


    """" save results """
    with open(os.path.join(op_dir, 'result.txt'), 'w') as f:
        f.write("Class: {}\nNumebr of Samples: {}\nMethod: {}\nStep Size: {}\nStratified K Fold: {}\nNumber of Features: {}\nRFE Time: {}s\nTrain Time: {}s\nTest Time: {}s\nAcc: {}\nTPR: {}\nFPR: {}\nTH: {}\n".format(
            class_name, n_samples, method, step, K, n_features, rfe_time, train_time, test_time, accuracy, TPR, FPR, th))

    with open(os.path.join(op_dir, 'model.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def test_model(selection, class_name, method, step, K, n_features, rfe_time):
    input_data = dataset.get_detector_dataset('val', Config.class_names)
    x_train, y_train, meta_train = input_data
    y_train = 1-y_train
    n_samples = len(x_train)
    if selection:
      x_train = x_train[:,:,:, selection]

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    ngpus = max(strategy.num_replicas_in_sync,1)
    print("Number of devices: {}".format(ngpus))
    train_data = tf.data.Dataset.from_tensor_slices(x_train)
    # The batch size must now be set on the Dataset objects.
    train_data = train_data.batch(128*ngpus)
    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)

    model_name = 'cnn_bn'
    model_file = './checkpoints/ndt_'+class_name+'_'+method.lower()+'_'+model_name+'.h5'
    train_time  = -1

    """" output directory name """
    op_dir = './data/model_results/'+class_name+'/'+method
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    op_name = 'Step:'+str(step)+'_'+'K_Fold:'+ str(K)+'_'+'Samples:'+str(n_samples)+'_'+ str(current_time)
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)
    op_dir = os.path.join(op_dir, op_name)
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    """" test the model """
    with strategy.scope():
        model = load_model_from_file(checkpoint_name=model_file)
    accuracy, TPR, FPR, th, test_time = test_trained_model_nn(model, class_name, method, op_dir, train_data, y_train, meta_train)


    """" save results """
    with open(os.path.join(op_dir, 'result.txt'), 'w') as f:
        f.write("Class: {}\nNumebr of Samples: {}\nMethod: {}\nStep Size: {}\nStratified K Fold: {}\nNumber of Features: {}\nRFE Time: {}s\nTrain Time: {}s\nTest Time: {}s\nAcc: {}\nTPR: {}\nFPR: {}\nTH: {}\n".format(
            class_name, n_samples, method, step, K, n_features, rfe_time, train_time, test_time, accuracy, TPR, FPR, th))

    with open(os.path.join(op_dir, 'model.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def RFE_with_sum_old():
    
    class_name = Config.cs_dir_name
    input_data = dataset.get_detector_dataset('val', Config.class_names)
    x_train, y_train, meta_train = input_data

    x_train, _ = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, _ = train_test_split(y_train, test_size=0.15, random_state=25)
    
    feature_array = x_train
    target = y_train
    target = np.array(target)
    target = target.flatten()
    feature_array = np.sum(np.sum(feature_array, axis=1), axis=1)
    
    # Use Recursive Feature Elimination to select the top 3 features
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    selector = RFECV(estimator=rf, cv=StratifiedKFold(5), scoring='accuracy', step = 4)
    
    # selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=32, step=4)
    selector = selector.fit(feature_array,target)
    # n_features = len(selector.scoring)
    #scores = selector.grid_scores_
    acc = selector.cv_results_
    # Plot the accuracies
    # plt.plot(range(1, n_features + 1), scoring_metric)
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Accuracy")
    # plt.title("Recursive Feature Elimination")
    # plt.show()
    final_selection = np.argwhere(selector.support_)[:,0].tolist()
    # # Find the indices of the highest number
    # # indices = [index for index, num in enumerate(scores) if num == 50]
    # # index = max(indices)
    # print("Accuracy:", scores)
    # #print("Highest Accuracy:", scores)
    # # print("Manual Highest Accuracy:", scores[index])
    # print("Index:", index)
    #Feature Reduction end
    split_test =[]
    for i in range(5): 
        split_test.append(acc[f'split{i}_test_score'])
    split_test=np.array(split_test)
    z=np.average(split_test,axis=0)
    print(z)
    best_index = np.argmax(acc['mean_test_score'])
    highest_accuracy = acc['mean_test_score'][best_index]
    #best_features = selector.grid_scores_[best_index].cv_validation_scores
    #best_features = np.array(feature_names)[rfecv.support_]
    # best_features = np.array(feature_array)[best_index].tolist()
    # print("Best Features:", best_features)
    print("Highest Accuracy:", highest_accuracy)
    print("Accuracy:", highest_accuracy)
    print("Best Features:", best_index)
    print("Scoring", acc)
 
    print(final_selection)
    return final_selection


def RFE_with_sum():
    start = time.time()
    method = 'RFE_with_summation'
    class_name = Config.cs_dir_name
    x_train, y_train, _ = dataset.get_detector_dataset('val', Config.class_names)
    n_samples = len(x_train)
    x_train, _ = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, _ = train_test_split(y_train, test_size=0.15, random_state=25)
   
    # # Calculate mean and standard deviation along the (0, 1) axes
    # channel_mean = np.mean(x_train, axis=(0, 1))
    # channel_std = np.std(x_train, axis=(0, 1))
    
    # # Normalize each channel independently
    # x_train_normalized = (x_train - channel_mean) / channel_std    

    feature_array = x_train
    target = y_train
    target = np.array(target)
    target = target.flatten()
    feature_array = np.sum(np.sum(feature_array, axis=1), axis=1)
    
    K = 5
    step = 1

    print("Running Recursive Elimation...")
    print("Method:", method)
    print("Class:", class_name)
    print("Step:", step)
    print("K:", K)

    # Use Recursive Feature Elimination to select the top 3 features
    rf = RandomForestClassifier(n_estimators=10)#, random_state=42)
    selector = RFECV(estimator=rf, cv=StratifiedKFold(K), scoring='accuracy', step = step)
    
    # selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=32, step=4)
    selector = selector.fit(feature_array,target)
    # n_features = len(selector.scoring)
    #scores = selector.grid_scores_
    acc = selector.cv_results_
    final_selection = np.argwhere(selector.support_)[:,0].tolist()
    # # Find the indices of the highest number
    # # indices = [index for index, num in enumerate(scores) if num == 50]
    # # index = max(indices)

    #Feature Reduction end
    split_test =[]
    for i in range(3): 
        split_test.append(acc[f'split{i}_test_score'])
    split_test=np.array(split_test)
    z = np.average(split_test,axis=0)
    best_index = np.argmax(acc['mean_test_score'])
    highest_accuracy = acc['mean_test_score'][best_index]
    # best_features = selector.grid_scores_[best_index].cv_validation_scores
    # best_features = np.array(feature_names)[rfecv.support_]
    # best_features = np.array(feature_array)[best_index].tolist()

    # print("Best Features:", best_features)
    # print("Highest Accuracy:", highest_accuracy)
    # print("Accuracy:", highest_accuracy)
    # print("Best Features:", best_index)
    # print("Scoring", acc)

    n_features = len(final_selection)

    end = time.time()
    elapsed_time = end - start	

    print("No. features:", n_features)
    print("Step size:", step)
    print("StratifiedKFold K:", K)
    print("Required Time in seconds:", elapsed_time)
    # print("Final:", final_selection)
    # print("Highest Accuracy:", highest_accuracy)
    # print("Index:", index)

    """ save selected feature list """
    op_dir = './data/rfe_results/'
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)
    op_file = op_dir+class_name+'_'+method+'.txt'
    np.savetxt(op_file, np.array(final_selection), fmt='%d')

    return final_selection, class_name, n_samples, method, step, K, n_features, elapsed_time

	
def recursive_feature_elimination():
    start = time.time()
    method = 'RFE_with_Secondary_Network'
    final_selection = []
    class_name = Config.cs_dir_name
    x_train, y_train, _ = dataset.get_detector_dataset('val', Config.class_names)
    n_samples = len(x_train)

    x_train, _ = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, _ = train_test_split(y_train, test_size=0.15, random_state=25)
    
    # Calculate mean and standard deviation along the (0, 1) axes
    channel_mean = np.mean(x_train, axis=(0, 1))
    channel_std = np.std(x_train, axis=(0, 1))
    
    # Normalize each channel independently
    x_train_normalized = (x_train - channel_mean) / channel_std
    
    # feat_min = np.amin(x_train,axis=(0,1,2))
    # feat_max = np.amax(x_train,axis=(0,1,2))
    # feat_min = np.reshape(feat_min,(1, 1, 1, 512))
    # feat_max = np.reshape(feat_max,(1, 1, 1, 512))
    
    # # normalize feature_array
    # x_train = (x_train - feat_min) / (feat_max - feat_min)

    model_name = 'cnn_bn'

    # reduce learning rate when the model stops improving
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(verbose=0, min_lr=0.0001, patience=10, factor=0.4)
    # stop training early when the model stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, verbose=0)
    callbacks_list = [learning_rate_dec, early_stopping]
    accuracy, eliminated_features = [], []
    index_array = np.arange(x_train.shape[-1])
    K = 5
    init_n_features, step = x_train.shape[-1], 1
    n_iter = int(init_n_features/step)

    print("Running Recursive Elimation...")
    print("Method:", method)
    print("Class:", class_name)
    print("Step:", step)
    print("Iterations:", n_iter)
    print("K:", K)

    for x in range(n_iter):
        skf = StratifiedKFold(K)
        first_layer_weights = k_accuracy = 0
        for k, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
            print("Iteration:",x, "Fold:", k)
            x_tr, y_tr = np.take(x_train, train_index, axis=0), np.take(y_train, train_index, axis=0)
            x_vl, y_vl = np.take(x_train, val_index, axis=0), np.take(y_train, val_index, axis=0)

            model = make_simple_model_seq(input_shape=x_tr[0].shape)
            model = compile_model(model, loss='binary_crossentropy')
            # # model.summary()
            
            history = model.fit(x_tr, y_tr, validation_data=(x_vl,y_vl), epochs=12, batch_size=50, callbacks=callbacks_list, verbose=0)
            first_layer_weights += np.squeeze(np.sum(np.absolute(model.layers[0].get_weights()[0]), axis=(0,1,3)))/K
            k_accuracy += history.history['val_accuracy'][-1]*100/K

            """ clean the memory """
            del x_tr
            del y_tr
            del x_vl
            del y_vl
            gc.collect()

        first_layer_sorted_indices = np.argsort(first_layer_weights)
        first_layer_sorted_indices = first_layer_sorted_indices[:step]
        eliminated_features.append(np.take(index_array,first_layer_sorted_indices))
        index_array = np.delete(index_array,first_layer_sorted_indices)
        accuracy.append(math.floor(k_accuracy))

        """ update x_train and clean the memory """
        x_train_updated = np.delete(x_train,first_layer_sorted_indices,3)
        del x_train
        gc.collect()
        x_train = x_train_updated

        print('Iteration:', x)
        print('Accuracy:', accuracy)
        print('Eliminated:', eliminated_features[-1])

    # Find the highest number
    highest_accuracy = max(accuracy)
    # plt.plot(accuracy)
    # plt.show()

    # Find the indices of the highest number
    indices = [index for index, num in enumerate(accuracy) if num == highest_accuracy]
    index = max(indices)
    
    for i in range(len(eliminated_features)):
        if i>=index:
            final_selection = final_selection + eliminated_features[i].tolist()
    final_selection.sort()
    end = time.time()
    elapsed_time = end - start

    n_features = len(final_selection)
    print("No. features:", n_features)
    print("Step size:", step)
    print("StratifiedKFold K:", K)
    print("Final:", final_selection)
    print("Highest Accuracy:", highest_accuracy)
    print("Index:", index)
    print("Required Time in seconds:", elapsed_time)

    """ save selected feature list """
    op_dir = './data/rfe_results/'
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)
    op_file = op_dir+class_name+'_'+method+'.txt'
    np.savetxt(op_file, np.array(final_selection), fmt='%d')
    
    return final_selection, class_name, n_samples, method, step, K, n_features, elapsed_time
 
	
def test_trained_model_nn(model, class_name, method, op_dir, x_val, y_val, meta_val):
    start = time.time()
    preds = model.predict(x_val)
    preds_05 = (preds > 0.5).astype(int)
    end = time.time()
    test_time = end - start

    accuracy = accuracy_measure(y_val, preds)
    cm = metrics.confusion_matrix(y_val, preds_05)
    plot_confusion_matrix_new(cm,  target_names=['False', 'True'], normalize=False,
                              title="Confusion Matrix (th={:.2f})".format(0.5), path=op_dir, name=class_name+"_"+ method.lower()+"_cm_acc")
    TPR, FPR, th, cm = fpr_at_95_tpr(y_val, preds)


    print("Accuracy: ")
    print("\t", f'{round(accuracy*100,2)}%',\
          "\n\t", f'{round(TPR*100,2)}%',\
          "\n\t", f'{round(FPR*100,2)}%',\
          "\n\t", f'{round(th,3)}')

    plot_confusion_matrix_new(cm,  target_names=['False', 'True'], normalize=False,
                              title="Confusion Matrix (th={:.2f})".format(th), path=op_dir, name=class_name+"_"+ method.lower()+"_cm_fpr")

    op_file = op_dir+f'/{class_name}'+'_'+method.lower()+'_predictions.txt'
    np.savetxt(op_file, np.column_stack((preds, y_val)), delimiter=' ', fmt='%3.6f')

    per_class_accuracy(class_name, method, op_dir, y_val, preds, meta_val)

    return accuracy, TPR, FPR, th, test_time


def per_class_accuracy(class_name, method, op_dir, y_val, preds, meta_val):
    n_class = len(Config.class_names);
    if n_class>1:
        meta_label = meta_val[:,-(2*n_class+1):].astype(int)
        op_file = op_dir+'/class_name'+'_'+method.lower()+'_per_class.txt'
        with open(op_file, 'w') as f:
            for i in range(n_class):
              y_select = y_val[meta_label[:,n_class+i]>0]
              pred_select = preds[meta_label[:,n_class+i]>0]
              n_samples = len(y_select)
              accuracy = accuracy_measure(y_select, pred_select)
    
              f.write("Class: {}\nNumebr of Samples: {}\nMethod: {}\nAcc: {}\nTH: {}\n\n".format(
                  Config.class_names[i], n_samples, method, accuracy, 0.5))


def new_model_512(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')

    x = keras.layers.Conv2D(256, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.Conv2D(512, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D((2,2))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(1024, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D((2,2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def new_model(input_shape=(13, 13, 1)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')

    x = keras.layers.Conv2D(64, (3, 3), padding="same",
                            activation='relu')(inputs)
    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D((2,2))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.MaxPool2D((2,2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model


def train_two_neuron_new():
    
    class_name = 'person'
    num = '1'

    x_train, y_train, meta_train = dataset.get_detector_dataset(
        'val', Config.class_names)

    x_val, y_val, meta_val = dataset.get_detector_dataset(
        'val', Config.class_names)


    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.5, random_state=25)

    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/new_dt_{}_{}_{}'.format(class_name, num, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=3, write_grads=True)
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1, min_lr=0.0001, patience=3, factor=0.5)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)

    callbacks_list = [learning_rate_dec,
                      tensorboard, early_stopping]
    model = new_model()
    model = compile_model(model)
    model.summary()
    model.fit(x_train, y_train, epochs=50, batch_size=128,
              validation_data=(x_val, y_val), callbacks=callbacks_list)

    model.save('./checkpoints/person_trained_non_aug.h5')


def train_tt():
    class_name = 'car'
    num = '1'
    input_data = dataset.get_detector_dataset('val', Config.class_names)
    x_train, y_train, meta_train = input_data
    x_val, y_val, meta_val = input_data
    x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, y_val = train_test_split(y_train, test_size=0.15, random_state=25)
    
	
    
    y_train = np.array(y_train)
    
    print(x_train.shape)
    y_train = y_train.flatten()
    print(y_train)
    # feature_array = np.sum(np.sum(feature_array, axis=1), axis=1)

    n = 50
    # Use Recursive Feature Elimination to select the top 3 features
    selector = RFE(estimator = make_3d_model(), n_features_to_select=n, step=1)
    selector = selector.fit(x_train,y_train)
    print("Feature ranking:", rfe.ranking_)
    x_train = x_train[:, rfe.support_]
    

    model_name = 'cnn_bn'

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/new_dt_'+model_name+'_'+class_name+num +
        '{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=3, write_grads=True)

    # reduce learning rate when the model stops improving
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1, min_lr=0.0001, patience=10, factor=0.4)

    # stop training early when the model stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=50, verbose=1)

    # save the best model based on validation accuracy
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        './checkpoints/ndt_person_'+model_name+'.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)

    callbacks_list = [learning_rate_dec, tensorboard,
                      early_stopping, model_checkpoint]

    #model = make_simple_model()
    model = make_simple_model_rfe()
    model = compile_model(model, loss='binary_crossentropy')

    model.summary()
    op_dir = './data/model_results/cnn_net1'
    with open(os.path.join(op_dir, 'model_6.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    model.fit(x_train, y_train, epochs=50, batch_size=50,
              validation_data=(x_val, y_val), callbacks=callbacks_list)

    # The best model has already been saved by ModelCheckpoint, so we don't need to save the model again here

    model=load_model_from_file(checkpoint_name='./checkpoints/ndt_person_cnn_bn.h5')
    test_trained_model_nn(model, model_name, x_val, y_val2=y_val)


def train_two_neuron():
    class_name = 'person'
    num = '1'

    x_train, y_train, meta_train = dataset.get_detector_dataset(
        'val', Config.class_names)

    x_val, y_val, meta_val = dataset.get_detector_dataset(
        'val', Config.class_names)
    x_train, x_val = train_test_split(x_train, test_size=0.5, random_state=25)
    
	
    # x_combined = np.concatenate((x_train, x_val))
    # y_combined = np.concatenate((y_train, y_val))
    # meta_combined = np.concatenate((meta_train, meta_val))

    # x_train, x_val, y_train, y_val = train_test_split(x_combined, y_combined,
    #                                                   stratify=y_combined,
    #                                                   test_size=0.2)

    # print(len(X_train), len(X_val))

    # exit(0)
    y_train, y_val = train_test_split(y_train, test_size=0.5, random_state=25)
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)

    print(y_val.shape)
    # y_val = y_val[:5000, :]

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/new_dt_'+class_name + num+'{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=3, write_grads=True)
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1, min_lr=0.001, patience=5, factor=0.5)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)

    callbacks_list = [learning_rate_dec,
                      tensorboard]
    # model = make_model_with_global_average_pooling_two_op()
    # model = make_simple_model_two_neurons()

    model = make_large_model_two_neurons()
    #model=make_simple_model()#make_network1_alt()

    # model = make_model_five_labels()
    # model = keras.models.load_model('./checkpoints/person_train_large_1.h5')

    model = compile_model(model)
    model.summary()
    model.fit(x_train, y_train, epochs=50, batch_size=128,
              validation_data=(x_val, y_val), callbacks=callbacks_list)
    predictions = model.predict(x_val)
    predicted_classes = np.argmax(predictions, axis=1)
    threshold = 0.7  # Set the threshold for high confidence prediction
    high_confidence_predictions = predictions[predictions > threshold]
    rounded_high_confidence_predictions = [round(i, 2) for i in high_confidence_predictions]
    high_confidence_indices = np.where(predictions > threshold)[0]
    high_confidence_classes = predicted_classes[high_confidence_indices.ravel()]

    high_confidence_results = np.stack((high_confidence_predictions, high_confidence_classes), axis=-1)

    print("High confidence predictions and classes:", high_confidence_results)

    model.save('./checkpoints/person_trained_non_aug.h5')
    # model_name='person_trained_non_aug.h5'
    # test_trained_model_nn(model, model_name, x_val, y_val2=y_val)

def train_two_neuron_ne():
    class_name = 'person'
    num = '1'

    x_train, y_train, meta_train = dataset.get_detector_dataset(
        'val', Config.class_names)

    x_val, y_val, meta_val = dataset.get_detector_dataset(
        'val', Config.class_names)

    x_train, x_val, y_train, y_val = train_test_split(
        np.concatenate((x_train, x_val)),
        np.concatenate((y_train, y_val)),
        stratify=np.concatenate((y_train, y_val)),
        test_size=0.25, random_state=25
    )

    # y_train = keras.utils.to_categorical(y_train)
    # y_val = keras.utils.to_categorical(y_val)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/new_dt_'+class_name + num+'{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=3, write_grads=True)
    learning_rate_dec = tf.keras.callbacks.ReduceLROnPlateau(
        verbose=1, min_lr=0.0001, patience=3, factor=0.5)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)

    callbacks_list = [learning_rate_dec,
                      tensorboard, early_stopping]

    model = make_large_model_two_neurons()
    model = compile_model(model)

    best_weights = None
    best_acc = 0.0
    for epoch in range(50):
        model.fit(x_train, y_train, epochs=50, batch_size=128,
                  validation_data=(x_val, y_val), callbacks=callbacks_list, shuffle=True)

        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.get_weights()
    
    model.set_weights(best_weights)
    model.save('./checkpoints/person_trained_non_aug.h5')


def load_model_from_file(checkpoint_name):
    model = keras.models.load_model(checkpoint_name)
    model.summary()
    return model


def test_trained_model(model):

    # x_val, y_val2, meta_val = dataset.get_detector_dataset(
        # 'val', Config.class_names)

    # y_val = keras.utils.to_categorical(y_val2)
    start_time = time.perf_counter()
    preds = model.predict(x_val)

    # preds = (preds > 0.5).astype(int)

    print(time.perf_counter() - start_time, "seconds --------------------")
    print(sum(preds)/len(preds))

    # preds = np.argmax(preds, axis=1)

    fpr, tpr, thresholds = metrics.roc_curve(
        np.array(y_val2), np.array(preds))

    auc = metrics.auc(fpr, tpr)
    print(auc)
    ind = np.argmax(tpr >= 0.95)
    th = thresholds[ind]
    fp = fpr[ind]
    print("the threshold is", th, " The fpr is", fp)
    plot_roc(fpr, tpr, thresholds, auc,
             class_name=' '.join(Config.class_names))
    preds = (preds > 0.5).astype(int)

    cm = metrics.confusion_matrix(y_val2, preds)

    plot_confusion_matrix(cm, ['0', '1', '2', '3', '4'],
                          normalize=False, class_name=' '.join(Config.class_names))
    print(cm)


def test_trained_model_new_haibo_model(modelfile):

    # modelfile = '/home/bijay/Dropbox/CESGM_project/Bijay/Code_aug02/checkpoints/person_single_output.h5'
    # model = keras.models.load_model(modelfile)

    modelfile = '/home/bijay/Dropbox/CESGM_project/Haibo/experiments/models/backup/testc1_cnn3_best.h5'
    modelfile = './checkpoints/testc1_cnn3_final_2.h5'

    # modelfile = './checkpoints/testc1_cnn3_final_2.h5'
    # modelfile = './checkpoints/testc1_cnn3_final_2.h5'

    model = tf.keras.models.load_model(modelfile, custom_objects={
        'LeakyReLU': actfunc})

    # data_path = '/home/bijay/Dropbox/CESGM_project/Bijay/DatasetWithNewCode/val/person/without_bbox.dat'

    data_path = '/media/bijay/Projects/Datasets/val/person/outputs/data.dat'
    x_val, y_val2, meta_val = pickle_load_save.load(data_path)

    xtest_tmp = np.split(x_val, 6, 3)

    xtest = []
    for k in range(3):
        xtest.append(xtest_tmp[k]*xtest_tmp[k+3])
    xtest_f = np.squeeze(np.stack((xtest[0], xtest[1], xtest[2]), axis=3))
    x_val = xtest_f

    preds = model.predict(x_val)

    preds_05 = (preds > 0.5).astype(int)
    print("Accuracy: ")
    print(accuracy_measure(y_val2, preds))

    cm = metrics.confusion_matrix(y_val2, preds_05)
    plot_confusion_matrix_new(cm,  target_names=['False', 'True'], normalize=False,
                              title="Confusion Matrix (th={:.2f})".format(0.5), path="./", name="cm_acc_net2")

    TPR, FPR, th, cm = fpr_at_95_tpr(y_val2, preds)
    print(TPR, "\n", FPR, "\n", th)
    plot_confusion_matrix_new(cm,  target_names=['False', 'True'], normalize=False,
                              title="Confusion Matrix (th={:.2f})".format(th), path="./", name="cm_fpr_net2")


def test_trained_model_new(modelfile):

    # modelfile = '/home/bijay/Dropbox/CESGM_project/Bijay/Code_aug02/checkpoints/person_single_output.h5'
    # model = keras.models.load_model(modelfile)

    # modelfile = '/home/bijay/Dropbox/CESGM_project/Haibo/experiments/models/backup/testc1_cnn3_best.h5'

    modelfile = '/home/local2/Ferdous/YOLO/checkpoints/person_trained_non_aug.h5'
    model = keras.models.load_model(modelfile)
    x_train, y_train, meta_train = dataset.get_detector_dataset(
        'val', Config.class_names)

    x_val, y_val2, meta_val = dataset.get_detector_dataset(
        'val', Config.class_names)
    x_train, x_val = train_test_split(x_train, test_size=0.5, random_state=25)
    y_train, y_val2 = train_test_split(y_train, test_size=0.5, random_state=25)
    # data_path = '/home/bijay/Dropbox/CESGM_project/Bijay/DatasetWithNewCode/val/person/without_bbox.dat'

    #data_path = '/home/local2/Ferdous/YOLO/Datasets/val/person/outputs/data.dat'
    #x_val, y_val2, meta_val = pickle_load_save.load(data_path)

    preds = model.predict(x_val)

    preds_05 = (preds > 0.5).astype(int)
    print("Accuracy: ")
    print(accuracy_measure(y_val2, preds))

    cm = metrics.confusion_matrix(y_val2, preds_05)
    plot_confusion_matrix_new(cm,  target_names=['False', 'True'], normalize=False,
                              title="Confusion Matrix (th={:.2f})".format(0.5), path="./", name="cm_acc")

    TPR, FPR, th, cm = fpr_at_95_tpr(y_val2, preds)
    print(TPR, "\n", FPR, "\n", th)
    plot_confusion_matrix_new(cm,  target_names=['False', 'True'], normalize=False,
                              title="Confusion Matrix (th={:.2f})".format(th), path="./", name="cm_fpr")


def accuracy_measure(testy, pred, thresh=0.5):
    thresholded = (pred > thresh).astype(int)
    cm = metrics.confusion_matrix(testy.flatten(), thresholded.flatten())
    TPR = cm[1, 1]/sum(cm[1])
    FPR = cm[0, 1]/sum(cm[0])
    PR = cm[1, 1]/sum(cm[:, 1])
    TNR = cm[0, 0]/sum(cm[0])
    FNR = cm[1, 0]/sum(cm[1])
    NPV = cm[0, 0]/sum(cm[:, 0])
    ACC = (cm[0, 0]+cm[1, 1])/sum(sum(cm))
    return ACC
    # return TPR, FPR, PR, TNR, FNR, NPV, ACC, thresh, cm

"""
FPR @ 95% TPR
"""


def fpr_at_95_tpr(testy, pred):
    thresh = 1
    res = 0.00001
    while(1):
        thresholded = (pred > thresh).astype(int)
        cm = metrics.confusion_matrix(testy.flatten(), thresholded.flatten())
        TPR = cm[1, 1]/sum(cm[1])
        FPR = cm[0, 1]/sum(cm[0])
        if(TPR >= 0.95):
            break
        thresh -= res

    return TPR, FPR, thresh, cm


def plot_confusion_matrix_new(cm, target_names, title='Confusion matrix', name='cm', path=None, cmap=None, normalize=True):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 20})

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    thresh = 1773
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=25)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=25)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig = plt.gcf()
    iname = os.path.join(path, name+'.png')
    fig.savefig(iname)
    print('CM Plot Saved to %s' % (iname))
    # plt.show()


def plot_roc(fpr, tpr, threshold, auc, class_name=''):
    plt.figure()
    lw = 2
    ind = np.argmax(tpr >= 0.95)
    th = threshold[ind]
    fp = fpr[ind]
    print("the threshold is", th, " The fpr is", fp)

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = {:0.4f}), TPR-95%-FPR= {:0.4f}'.format(auc, fp))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

    plt.savefig('/home/bijay/Dropbox/GM_Own/'+class_name+'_roc.png')

    plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, class_name='none'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.savefig('/home/bijay/Dropbox/GM_Own/jul29/'+class_name+'_cm.png')
    plt.show()


def get_cam():
    model = keras.models.load_model(
        './checkpoints/person_val_large.h5')

    model.summary()

    class_name = 'person'
    bbox = 'without_bb'
    num = '12'
    data_file = class_name+'_data_'+bbox

    x_val, y_val, meta_val = dataset.get_detector_dataset(
        'val', Config.class_names)
    y_val_orig = y_val
    y_val = keras.utils.to_categorical(y_val)

    gap_weights = model.layers[-1].get_weights()[0]

    print(gap_weights.shape)

    cam_model = keras.models.Model(inputs=model.input, outputs=(
        model.layers[-3].output, model.layers[-1].output))
    features, results = cam_model.predict(x_val)

    print(features.shape)
    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    pic_limit = 10

    for idx in range(8000):
        features_for_one_img = features[idx, :, :, :]
        file_id = meta_val[idx][3]
        print(meta_val[idx])
        file_path = os.path.join(Config.root_images_dir, file_id + '.jpg')
        pred = np.argmax(results[idx])
        orig = np.argmax(y_val[idx])

        if(true_true >= pic_limit and true_false >= pic_limit and false_false >= pic_limit and false_true >= pic_limit):
            break

        if(pred == 0 and orig == 0):
            if(false_false >= pic_limit):
                continue
            else:
                false_false += 1
        elif(pred == 1 and orig == 1):
            if(true_true >= pic_limit):
                continue
            else:
                true_true += 1
        elif(pred == 0 and orig == 1):
            if(false_true >= pic_limit):
                continue
            else:
                false_true += 1
        elif(pred == 1 and orig == 0):
            if(true_false >= pic_limit):
                continue
            else:
                true_false += 1

        # img = cv2.imread(file_path)[:, :, ::-1]
        # (h, w) = img.shape[:2]
        # img = cv2.resize(img, (416, 416))
        img = draw_image.draw_img_test_file(file_id)

        height_roomout = 416.0/features_for_one_img.shape[0]
        width_roomout = 416.0/features_for_one_img.shape[1]
        # print(height_roomout, width_roomout)
        # (results > 0.5).astype(int)

        # cam_features = features_for_one_img
        # cam_features = sp.ndimage.zoom(
        #     features_for_one_img, (height_roomout, width_roomout, 1), order=2)
        cam_features = cv2.resize(features_for_one_img, (416, 416))

        plt.figure(facecolor='white')
        cam_weights = gap_weights[:, pred]
        cam_output = np.dot(cam_features, cam_weights)
        # print(features_for_one_img.shape)

        buf = 'True Class = '+str(y_val_orig[idx][0]) + ', Predicted Class = ' + \
            str(pred) + ', Probability = ' + str(results[idx][pred])

        plt.figure()
        plt.xlabel(buf)
        plt.xticks(np.arange(0, 416, step=32), range(13))
        plt.yticks(np.arange(0, 416, step=32), reversed(range(13)))

        plt.imshow(img, alpha=0.7)
        plt.imshow(cam_output, cmap='jet', alpha=0.4)
        plt.grid(linestyle='-.', linewidth=0.5)
        # cam_output = cv2.applyColorMap(
        #     np.uint8(255 * cam_output), cv2.COLORMAP_VIRIDIS)

        cv2.imwrite(
            './images/orig/{}_{}_{}_{}.png'.format(orig, pred, idx, file_id), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        plt.savefig('./images/im/{}_{}_{}.png'.format(orig, pred, idx))
        plt.close()

        plt.figure()
        plt.xticks(np.arange(0, 416, step=32), range(13))
        plt.yticks(np.arange(0, 416, step=32), reversed(range(13)))
        plt.imshow(cam_output, cmap='jet')
        plt.grid()
        plt.savefig(
            './images/cam/{}_{}_{}.png'.format(orig, pred, idx))
        plt.close()
        # plt.show()

def make_cam_id(idx):
    model = keras.models.load_model(
        './checkpoints/person_val_ll_large.h5')
    # './checkpoints/person_trained_non_aug.h5')

    class_name = 'person'
    bbox = 'without_bb'
    num = '12'
    data_file = class_name+'_data_'+bbox

    x_val, y_val, meta_val = dataset.get_detector_dataset(
        'val', Config.class_names)
    y_val_orig = y_val
    y_val = keras.utils.to_categorical(y_val)
    results = model.predict(x_val)

    idx = idx
    file_id = meta_val[idx][3]
    print(meta_val[idx])
    file_path = os.path.join(Config.root_images_dir, file_id + '.jpg')
    pred = np.argmax(results[idx])
    orig = np.argmax(y_val[idx])

    img = draw_image.draw_img_test_file(file_id)

    plt.figure(facecolor='white')
    cam = GradCAM(model, pred, layerName='conv2d')

    img_pred = np.expand_dims(x_val[idx], axis=0)
    cam_output, orig_cam = cam.compute_heatmap_2(img_pred)

    heatmap = orig_cam
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + 1e-5
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    print(heatmap)
    print(heatmap.shape)
    print(np.min(heatmap))
    print(np.max(heatmap))
    # print(features_for_one_img.shape)

    buf = 'True Class = '+str(y_val_orig[idx][0]) + ', Predicted Class = ' + \
        str(pred) + ', Probability = ' + str(results[idx][pred])

    fig, ax = plt.subplots()
    # plt.figure()
    # turn off the frame
    ax.set_frame_on(False)
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(0, 416, step=32), minor=False)
    ax.set_xticks(np.arange(0, 416, step=32),  minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    labels = range(13)
    # note I could have used nba_sort.columns but made "labels" instead
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)

    plt.xlabel(buf)
    # plt.xticks(np.arange(0, 416, step=32), range(13))
    # plt.yticks(np.arange(0, 416, step=32), reversed(range(13)))

    plt.imshow(img, alpha=0.7)
    plt.imshow(cam_output, cmap='jet', alpha=0.4)
    plt.grid(linestyle='-.', linewidth=0.5)
    # cam_output = cv2.applyColorMap(
    #     np.uint8(255 * cam_output), cv2.COLORMAP_VIRIDIS)

    if not os.path.exists('./images/orig'):
        os.makedirs('./images/orig')

    if not os.path.exists('./images/im'):
        os.makedirs('./images/im')

    if not os.path.exists('./images/cam'):
        os.makedirs('./images/cam')

    plt.savefig('./images/im/BBBBB_{}_{}_{}.png'.format(orig, pred,
                                                        idx), bbox_inches='tight', pad_inches=0)
    plt.close()

    fig, ax = plt.subplots()
    # plt.figure()
    # turn off the frame
    ax.set_frame_on(False)
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(0, 416, step=32), minor=False)
    ax.set_xticks(np.arange(0, 416, step=32),  minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    labels = range(13)
    # note I could have used nba_sort.columns but made "labels" instead
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)

    # plt.figure()
    # plt.xticks(np.arange(0, 416, step=32), range(13))
    # plt.yticks(np.arange(0, 416, step=32), reversed(range(13)))
    plt.imshow(img)
    plt.grid()
    plt.savefig(
        './images/orig/BBBBB_{}_{}_{}_{}.png'.format(orig, pred, idx, file_id), bbox_inches='tight', pad_inches=0)
    plt.close()

    # cv2.imwrite(
    #     './images/orig/{}_{}_{}_{}.png'.format(orig, pred, idx, file_id), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    fig, ax = plt.subplots()
    # plt.figure()
    # turn off the frame
    ax.set_frame_on(False)
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(0, 416, step=32), minor=False)
    ax.set_xticks(np.arange(0, 416, step=32),  minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    labels = range(13)
    # note I could have used nba_sort.columns but made "labels" instead
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)

    # plt.figure()
    # plt.xticks(np.arange(0, 416, step=32), range(13))
    # plt.yticks(np.arange(0, 416, step=32), reversed(range(13)))
    plt.imshow(cam_output, cmap='jet')
    plt.grid()
    plt.savefig(
        './images/cam/BBBBB_{}_{}_{}.png'.format(orig, pred, idx), bbox_inches='tight', pad_inches=0)
    plt.close()

    return heatmap


def get_cam_all():
    model = keras.models.load_model(
        './checkpoints/person_val_ll_large.h5')

    model.summary()

    # return
    class_name = 'person'
    bbox = 'without_bb'
    num = '12'
    data_file = class_name+'_data_'+bbox

    x_val, y_val, meta_val = dataset.get_detector_dataset(
        'val', Config.class_names)
    y_val_orig = y_val
    y_val = keras.utils.to_categorical(y_val)
    results = model.predict(x_val)

    true_true = 0
    true_false = 0
    false_true = 0
    false_false = 0
    pic_limit = 20

    for idx in range(8000):
        file_id = meta_val[idx][3]
        print(meta_val[idx])
        file_path = os.path.join(Config.root_images_dir, file_id + '.jpg')
        pred = np.argmax(results[idx])
        orig = np.argmax(y_val[idx])

        # if(true_false >= pic_limit):
        #     break

        if(true_true >= pic_limit and true_false >= pic_limit and false_false >= pic_limit and false_true >= pic_limit):
            break

        if(pred == 0 and orig == 0):
            if(false_false >= pic_limit):
                continue
            else:
                false_false += 1
        else:
            continue

        # elif(pred == 1 and orig == 1):
        #     if(true_true >= pic_limit):
        #         continue
        #     else:
        #         true_true += 1
        # elif(pred == 0 and orig == 1):
        #     if(false_true >= pic_limit):
        #         continue
        #     else:
        #         false_true += 1
        # elif(pred == 1 and orig == 0):
        #     if(true_false >= pic_limit):
        #         continue
        #     else:
        #         true_false += 1
        # else:
        #     continue

        # img = cv2.imread(file_path)[:, :, ::-1]
        # (h, w) = img.shape[:2]
        # img = cv2.resize(img, (416, 416))
        Config.aug_dir_name = 'no_aug/person'

        img = draw_image.draw_img_test_file(file_id)

        # height_roomout = 416.0/features_for_one_img.shape[0]
        # width_roomout = 416.0/features_for_one_img.shape[1]
        # print(height_roomout, width_roomout)
        # (results > 0.5).astype(int)

        # cam_features = features_for_one_img
        # cam_features = sp.ndimage.zoom(
        #     features_for_one_img, (height_roomout, width_roomout, 1), order=2)
        # cam_features = cv2.resize(features_for_one_img, (416, 416))

        plt.figure(facecolor='white')
        cam = GradCAM(model, pred, layerName='conv2d')

        img_pred = np.expand_dims(x_val[idx], axis=0)
        cam_output, orig_cam = cam.compute_heatmap_2(img_pred)

        # print(features_for_one_img.shape)

        buf = 'True Class = '+str(y_val_orig[idx][0]) + ', Predicted Class = ' + \
            str(pred) + ', Probability = ' + str(results[idx][pred])

        fig, ax = plt.subplots()
        # plt.figure()
        # turn off the frame
        ax.set_frame_on(False)
        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(0, 416, step=32), minor=False)
        ax.set_xticks(np.arange(0, 416, step=32),  minor=False)
        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        labels = range(13)
        # note I could have used nba_sort.columns but made "labels" instead
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(labels, minor=False)

        plt.xlabel(buf)
        # plt.xticks(np.arange(0, 416, step=32), range(13))
        # plt.yticks(np.arange(0, 416, step=32), reversed(range(13)))

        plt.imshow(img, alpha=0.7)
        plt.imshow(cam_output, cmap='jet', alpha=0.4)
        plt.grid(linestyle='-.', linewidth=0.5)

        # cam_output = cv2.applyColorMap(
        #     np.uint8(255 * cam_output), cv2.COLORMAP_VIRIDIS)

        if not os.path.exists('./images/orig'):
            os.makedirs('./images/orig')

        if not os.path.exists('./images/orig2'):
            os.makedirs('./images/orig2')

        if not os.path.exists('./images/im'):
            os.makedirs('./images/im')

        if not os.path.exists('./images/cam'):
            os.makedirs('./images/cam')

        plt.savefig('./images/im/{}_{}_{}.png'.format(orig, pred,
                                                      idx), bbox_inches='tight', pad_inches=0)
        plt.close()

        fig, ax = plt.subplots()
        # plt.figure()
        # turn off the frame
        ax.set_frame_on(False)
        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(0, 416, step=32), minor=False)
        ax.set_xticks(np.arange(0, 416, step=32),  minor=False)
        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        labels = range(13)
        # note I could have used nba_sort.columns but made "labels" instead
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(labels, minor=False)

        # plt.figure()
        # plt.xticks(np.arange(0, 416, step=32), range(13))
        # plt.yticks(np.arange(0, 416, step=32), reversed(range(13)))
        plt.imshow(img)
        plt.grid()
        plt.savefig(
            './images/orig/{}_{}_{}_{}.png'.format(orig, pred, idx, file_id), bbox_inches='tight', pad_inches=0)
        plt.close()

        # cv2.imwrite(
        #     './images/orig/{}_{}_{}_{}.png'.format(orig, pred, idx, file_id), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        fig, ax = plt.subplots()
        # plt.figure()
        # turn off the frame
        ax.set_frame_on(False)
        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(0, 416, step=32), minor=False)
        ax.set_xticks(np.arange(0, 416, step=32),  minor=False)
        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        labels = range(13)
        # note I could have used nba_sort.columns but made "labels" instead
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(labels, minor=False)

        # plt.figure()
        # plt.xticks(np.arange(0, 416, step=32), range(13))
        # plt.yticks(np.arange(0, 416, step=32), reversed(range(13)))
        plt.imshow(cam_output, cmap='jet')
        plt.grid()
        plt.savefig(
            './images/cam/{}_{}_{}.png'.format(orig, pred, idx), bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.show()

        shutil.copyfile(file_path, './images/orig2/'+file_id + '.jpg')


def grad_cam():
    model = keras.models.load_model(
        './checkpoints/person_val_ll_large.h5')

    model.summary()
    x_val, y_val, meta_val = dataset.get_detector_dataset(
        'val', Config.class_names)
    y_val_orig = y_val
    y_val = keras.utils.to_categorical(y_val)
    results = model.predict(x_val)

    print(results[0])
    label = np.argmax(results[0])

    file_id = meta_val[0][3]
    print(meta_val[0])
    file_path = os.path.join(Config.root_images_dir, file_id + '.jpg')
    orig = cv2.imread(file_path)
    resized = cv2.resize(orig, (224, 224))

    # load the input image from disk (in Keras/TensorFlow format) and
    # preprocess it
    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    cam = GradCAM(model, np.argmax(results[0]), layerName='conv2d')
    heatmap = cam.compute_heatmap_2(x_val[:1])

    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # draw the predicted label on the output image
    # cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    # cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.8, (255, 255, 255), 2)

    cv2.imshow("Output", output)
    cv2.waitKey(0)


def albation():
    # maximum = max(cam)
    # minimum = min(cam)
    x, y, meta = dataset.get_detector_dataset(
        'val', Config.class_names)

    model = keras.models.load_model(
        '/home/bijay/Dropbox/CESGM_project/Bijay/Code_aug02/checkpoints/person_single_output.h5')
    predictions = model.predict(x)
    predictions = [int(p >= 0.5) for p in predictions]

    print(y[0][0])

    true_predictions = [predictions[i] == y[i][0]
                        for i in range(len(predictions))]
    accuracy = sum(true_predictions) / len(true_predictions)
    print(accuracy)

    x[:, :, :, 3:6] = 0
    predictions = model.predict(x)
    predictions = [int(p >= 0.5) for p in predictions]
    true_predictions = [predictions[i] == y[i][0]
                        for i in range(len(predictions))]
    accuracy = sum(true_predictions) / len(true_predictions)
    print(accuracy)

    # model = keras.models.load_model('./checkpoints/person_single_output_2.h5')
    # print(x)


def test_data():
    x_train, y_train, meta_train = dataset.get_detector_dataset(
        'train', Config.class_names)

    pickle_load_save.load(custom_config.detector_data_file)

    print(len(meta_train))
    img_name = 'COCO_train2014_000000000077'

    for i, f in enumerate(meta_train):
        if(f[3] == img_name):
            print(i)
            # heatmap = make_cam_id(i)
            dt = x_train[i]
            dt = dt.reshape(13, 13, 3, 2)
            print(dt.shape)

    dt_f = dt[2:5, 2:5, :, :]
    count = 0
    dt = dt.reshape(13, 13, 6)
    class_plot = dt[:, :, 3]
    objectness_plot = dt[:, :, 0]

    make_confusion_matrix_type_heatmap(
        objectness_plot, plt_name='Objectness_0')
    make_confusion_matrix_type_heatmap(class_plot, plt_name='Class_prob_0')
    # make_confusion_matrix_type_heatmap(
    #     heatmap, valfmt="{x:.0f}", plt_name='Gradiant')

def make_confusion_matrix_type_heatmap(input_array, valfmt="{x:.1f}", plt_name='plt'):
    fig, ax = plt.subplots()

    labels = [i for i in range(0, 13)]
    im, cbar = draw_image.heatmap(input_array, labels, labels, ax=ax,
                                  cmap="cool")
    # , cbarlabel="harvest [t/year]")
    texts = draw_image.annotate_heatmap(im, valfmt=valfmt)

    fig.tight_layout()
    plt.savefig(plt_name+'.png')
    plt.savefig(plt_name+'.pdf')
    plt.show()


def make_confusion_matrix_type_heatmap_with_rect(input_array, valfmt="{x:.1f}", plt_name='plt', rect=[]):
    fig, ax = plt.subplots()

    labels = [i for i in range(0, 13)]
    im, cbar = draw_image.heatmap(input_array, labels, labels, ax=ax,
                                  cmap="cool")
    # , cbarlabel="harvest [t/year]")
    texts = draw_image.annotate_heatmap(im, valfmt=valfmt)

    if(len(rect) > 0):
        h = rect[3] - rect[1] + 0.8
        w = rect[2] - rect[0] + 0.8
        point = (float(rect[0])-0.4, float(rect[1])-0.4)
        rect1 = patches.Rectangle(
            point, w, h, linewidth=3, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect1)

    fig.tight_layout()
    plt.savefig(plt_name+'.png')
    # plt.savefig(plt_name+'.pdf')
    plt.show()


if __name__ == "__main__":
     #test_trained_model_new(None)
    
    # test_trained_model_new_haibo_model(None)

    # model = keras.models.load_model(
    #     '/home/bijay/Dropbox/CESGM_project/Bijay/Code_aug02/checkpoints/person_single_output.h5')
    # model.summary()

    # test_data()

    runconf = ['sum', 'secondary', 'all', 'sum', 'run_all'][4]
    runrfe = False

    for c in [['chair']]:#[['cup']], ['car'], ['chair'], ['person'], ['person', 'car'], ['person', 'car', 'chair'], ['person', 'car', 'chair', 'cup']]:
      Config.class_names = c
      Config.reset_all_vars()
      for i in range(0,4,1):
          if (i==0 or i==3) and (runconf=='sum' or runconf=='run_all'):
              if runrfe:
                ''' RFE with sum '''
                final_selection, class_name, n_samples, method, step, K, n_features, rfe_time = RFE_with_sum()
              else:
                ''' use previous RFE results '''
                class_name, method, step, K, rfe_time = Config.cs_dir_name, 'RFE_with_summation', 1, 5, -1
                op_file = './data/rfe_results/'+class_name+'_'+method+'.txt'
                final_selection = np.loadtxt(op_file).astype(int).tolist()
                n_features = len(final_selection)
              test_model(final_selection, class_name, method, step, K, n_features, rfe_time)
          elif i==1 and (runconf=='secondary' or runconf=='run_all'):
              if runrfe:
                ''' RFE with secondary network '''
                final_selection, class_name, n_samples, method, step, K, n_features, rfe_time = recursive_feature_elimination()
              else:
                ''' use previous RFE results '''
                class_name, method, step, K, rfe_time = Config.cs_dir_name, 'RFE_with_Secondary_Network', 1, 5, -1
                op_file = './data/rfe_results/'+class_name+'_'+method+'.txt'
                final_selection = np.loadtxt(op_file).astype(int).tolist()
                n_features = len(final_selection)
              test_model(final_selection, class_name, method, step, K, n_features, rfe_time)
          elif i==2 and (runconf=='all' or runconf=='run_all'):
            ''' All Features '''
            final_selection, class_name, n_samples, method, step, K,n_features, rfe_time = None, Config.cs_dir_name, 0, 'All_Features', 0, 0, 512, 0
            test_model(final_selection, class_name, method, step, K, n_features, rfe_time)