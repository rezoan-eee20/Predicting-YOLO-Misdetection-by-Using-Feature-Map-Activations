import os
import os.path as path
from config import Config, ConfigDTS
import logging
import csv
import pandas as pd
import json
from sklearn.utils import shuffle
import pickle_load_save
import tqdm
import numpy as np
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import random
import tensorflow as tf
######################################################
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
# actfunc = 'relu'
actfunc = tf.keras.layers.LeakyReLU(alpha=0.1)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import random
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
################################################DATASET_START#########################################################
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def create_csv_of_all_detected_files(csv_file=None):

    Config.reset_all_vars()

    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data.csv')

    lines = 0
    all_data = []

    stat = {}

    field_names = ['fid', 'file_name', 'label',
                   'augmentation', 'class_name', 'dataset']

    all_class_name_combinations = Config.get_all_class_names_combinations()

    output_dir_subfolder = [f.name for f in os.scandir(
        Config.cs_output_root) if f.is_dir()]

    for sub_folder in output_dir_subfolder:
        for cls_name in all_class_name_combinations:
            Config.aug_dir_name = os.path.join(sub_folder, cls_name)
            if not os.path.exists(Config.cs_output_root):
                logging.warn('Not found: Ignoring: {}'.format(
                    Config.aug_dir_name))
                continue

            logging.info('found: processing: {}'.format(
                Config.aug_dir_name))

            label_1_source = path.join(
                Config.cs_file_list_dir, 'l1_with_detections.txt')
            label_0_source = path.join(Config.cs_file_list_dir, 'l0.txt')
            label1 = file_lines_to_list(label_1_source)
            label0 = file_lines_to_list(label_0_source)
            
            lines += len(label1)
            lines += len(label0)
            data_dict = {}

            if cls_name not in stat:
                stat[cls_name] = {0: len(label0), 1: len(label1)}
            else:
                stat[cls_name][0] += len(label0)
                stat[cls_name][1] += len(label1)

            # if(len(label1) < len(label0)):
            #     print('More label 0')
            #     label0 = label0[:len(label1)]
            # else:
            #     label1 = label1[:len(label0)]
            #     print('more label 1')

            data_dict['augmentation'] = sub_folder
            data_dict['class_name'] = cls_name
            data_dict['dataset'] = Config.dt  # Dataset type

            for x in label1:
                data_dict['file_name'] = x.split('.')[0]
                data_dict['label'] = 1
                all_data.append(data_dict.copy())

            for x in label0:
                data_dict['file_name'] = x.split('.')[0]
                data_dict['label'] = 0
                all_data.append(data_dict.copy())

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for i, data in enumerate(all_data):
            data['fid'] = i+1
            writer.writerow(data)

    json_file_path = csv_file+'_count.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(stat, json_file)

    logging.info('Saved csv to {0} with {1} items.\nStat file in {2}'.format(
        csv_file, lines, json_file_path))


def create_new_csv_of_all_detected_files(csv_file=None):

    Config.reset_all_vars()

    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data.csv')

    lines = 0
    all_data = []

    stat = {}

    field_names = ['fid', 'file_name', 'label',
                   'augmentation', 'class_name', 'dataset']

    all_class_name_combinations = Config.get_all_class_names_combinations()

    output_dir_subfolder = [f.name for f in os.scandir(
        Config.cs_output_root) if f.is_dir()]

    for sub_folder in output_dir_subfolder:
        for cls_name in all_class_name_combinations:
            Config.aug_dir_name = os.path.join(sub_folder, cls_name)
            if not os.path.exists(Config.cs_output_root):
                logging.warn('Not found: Ignoring: {}'.format(
                    Config.aug_dir_name))
                continue

            logging.info('found: processing: {}'.format(
                Config.aug_dir_name))

            # Label 0 - all detection is correct - there are object and network detected it
            # label 1 -

            # a) Label 0 - there is person and the network detects it
            # b) Label 1 - there is person and the network fails to detect it
            # c) Label 2 - there is no person and the network detection is right
            # d) Label 4 - there is no person and the network detection is wrong
            # e) label 3 - network detects only some

            label_0_source = path.join(
                Config.cs_file_list_dir, 'l1_with_detections.txt')
            label_1_source = path.join(
                Config.cs_file_list_dir, 'l0_missed_all.txt')

            label_2_source = path.join(
                Config.cs_file_list_dir, 'l1_without_detections.txt')
            label_3_source = path.join(
                Config.cs_file_list_dir, 'l0_missed_some.txt')

            label_4_source = path.join(Config.cs_file_list_dir, 'l0_fp.txt')

            # label_0_source = path.join(Config.cs_file_list_dir, 'l0.txt')
            label0 = file_lines_to_list(label_0_source)
            label1 = file_lines_to_list(label_1_source)
            label2 = file_lines_to_list(label_2_source)
            label3 = file_lines_to_list(label_3_source)
            label4 = file_lines_to_list(label_4_source)

            label_array = [label0, label1, label2, label3, label4]

            lines += len(label1) + len(label0) + \
                len(label2) + len(label3) + len(label4)

            data_dict = {}

            if cls_name not in stat:
                stat[cls_name] = {0: len(label0), 1: len(label1), 2: len(
                    label2), 3: len(label3), 4: len(label4)}
            else:
                stat[cls_name][0] += len(label0)
                stat[cls_name][1] += len(label1)
                stat[cls_name][2] += len(label2)
                stat[cls_name][3] += len(label3)
                stat[cls_name][4] += len(label4)

            # if(len(label1) < len(label0)):
            #     print('More label 0')
            #     label0 = label0[:len(label1)]
            # else:
            #     label1 = label1[:len(label0)]
            #     print('more label 1')

            data_dict['augmentation'] = sub_folder
            data_dict['class_name'] = cls_name
            data_dict['dataset'] = Config.dt  # Dataset type

            for index, labels in enumerate(label_array):
                for x in labels:
                    data_dict['file_name'] = x.split('.')[0]
                    data_dict['label'] = index
                    all_data.append(data_dict.copy())

            # for x in label1:
            #     data_dict['file_name'] = x.split('.')[0]
            #     data_dict['label'] = 1
            #     all_data.append(data_dict.copy())

            # for x in label0:
            #     data_dict['file_name'] = x.split('.')[0]
            #     data_dict['label'] = 0
            #     all_data.append(data_dict.copy())

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for i, data in enumerate(all_data):
            data['fid'] = i+1
            writer.writerow(data)

    json_file_path = csv_file+'_count.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(stat, json_file)

    logging.info('Saved csv to {0} with {1} items.\nStat file in {2}'.format(
        csv_file, lines, json_file_path))


def create_dataset_csv(csv_file=None, randomize=True, output_csv=None):
    """This tries to create a balanced file list"""

    Config.reset_all_vars()

    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data.csv')

    if output_csv is None:
        output_csv = path.join(Config.cs_output_root, 'all_data_balanced.csv')

    print('The output root is {}'.format(Config.cs_output_root))
    count_file = path.join(Config.cs_output_root, 'all_data.csv_count.json')

    with open(count_file, 'r') as json_file:
        stat = json.load(json_file)

    all_class_names = Config.get_all_class_names_combinations()

    data = pd.read_csv(csv_file, index_col=0)
    all_data_coll = pd.DataFrame(columns=data.columns)

    for cls_name in all_class_names:
        cnts = stat[cls_name]
        print(cnts)
        label_cls = data.class_name == cls_name
        label0 = data[(data.class_name == cls_name) & (data.label == 0)]
        label1 = data[(data.class_name == cls_name) & (data.label == 1)]
        label2 = data[(data.class_name == cls_name) & (data.label == 2)]
        label3 = data[(data.class_name == cls_name) & (data.label == 3)]
        label4 = data[(data.class_name == cls_name) & (data.label == 4)]

        min_dict = min(cnts.values())
        print("*****",min_dict)

        label0 = label0[:min_dict]
        label1 = label1[:min_dict]
        label2 = label2[:min_dict]
        label3 = label3[:min_dict]
        label4 = label4[:min_dict]

        # if(cnts['0'] > cnts['1']):
        #     if randomize:
        #         label0 = shuffle(label0)
        #     label0 = label0[:cnts['1']]
        # else:
        #     if randomize:
        #         label1 = shuffle(label1)
        #     label1 = label1[:cnts['0']]

        all_data_coll = pd.concat(
            [all_data_coll, label0, label1, label2, label3, label4])

    all_data_coll.to_csv(output_csv)
    logging.info('Saved to {}'.format(output_csv))


def get_data_path(aug, class_name, file_name):
    file_name=str(file_name)
    return os.path.join(Config.cs_output_root, Config._output_data_dir_name, file_name+ '.data')
def get_gt_path(aug, class_name, file_name):
    file_name=str(file_name)
    return os.path.join(Config.cs_output_root, Config._gt_dir_name, file_name+ '.txt')

def create_dataset_pickle_old(csv_file=None, op_file_name='data.dat', include_bbox=False):
    """Saves a tuple of x,y and meta"""

    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data_balanced.csv')
    x = []
    y = []
    meta = []

    output_file_name = path.join(Config.cs_output_root, op_file_name)

    df = pd.read_csv(csv_file, index_col=0)
    print(df)

    df['data_path'] = df.apply(lambda x: get_data_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)

    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)

        x.append(read_x(row.data_path, include_bbox=include_bbox))
        y.append(row.label)
        meta.append((row.dataset, row.augmentation,
                     row.class_name, row.file_name))

    pickle_load_save.save(output_file_name, (np.array(x),
                                             np.array(y).reshape((-1, 1)), np.array(meta)))
    logging.info('Dataset saved to {}'.format(output_file_name))

    return output_file_name


def create_dataset_pickle(csv_file=None, op_file_name='data.dat', include_bbox=False):
    """Saves a tuple of x,y and meta"""

    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data_balanced.csv')
    # if csv_file is None:
        # csv_file = path.join(Config.cs_output_root, 'all_data.csv')
    x = []
    y = []
    meta = []

    output_file_name = path.join(Config.cs_output_root, op_file_name)

    df = pd.read_csv(csv_file, index_col=0)
    #print("DF:",df['label'])

    df['data_path'] = df.apply(lambda x: get_data_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)
    df['gt_path'] = df.apply(lambda x: get_gt_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)     
    
    feature_array = []
    bbox_array = []
    gt_array = []
    
    
    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)
        y.append(row.label)
        feature_array.append(read_x_features(row.data_path, include_bbox=include_bbox))
        bbox_array.append(read_x_bbox(row.gt_path, row.class_name))
        #print("row.gt_path, bbox_array:", row.gt_path, bbox_array)
    bbox_array = np.array(bbox_array) 
    
    feature_array = np.array(feature_array)
    print(feature_array.shape)
    #select=reduction(feature_array, y, 50)
    #select=reduction(feature_array, bbox_array, y, 50)
    #select = [1, 5, 14, 59, 62, 70, 71, 73, 74, 76, 103, 107, 110, 121, 126, 127, 130, 137, 141, 152, 173, 189, 195, 199, 220, 235, 243, 247, 257, 283, 291, 308, 313, 327, 332, 334, 366, 370, 380, 381, 387, 389, 404, 407, 462, 482, 488, 493, 507, 511]
    #[1, 5, 14, 59, 62, 64, 70, 73, 74, 76, 103, 117, 121, 126, 127, 133, 137, 151, 171, 173, 189, 199, 220, 235, 243, 247, 257, 283, 291, 299, 308, 313, 327, 328, 331, 332, 366, 370, 380, 381, 387, 389, 404, 407, 432, 452, 481, 493, 507, 511]
    #select = [1, 5, 14, 59, 62, 64, 70, 71, 73, 74, 76, 103, 107, 110, 121, 126, 127, 137, 151, 152, 173, 199, 220, 235, 243, 247, 257, 266, 283, 287, 291, 308, 313, 332, 334, 366, 370, 380, 381, 387, 389, 404, 407, 430, 431, 432, 462, 488, 493, 511]
    select = train_bn()

	#25select = [5, 14, 59, 62, 73, 74, 76, 107, 121, 126, 127, 139, 199, 220, 247, 257, 283, 291, 308, 332, 381, 387, 404, 407, 432]
    #select = [14, 59, 73, 74, 76, 126, 127, 220, 247, 257, 308, 332, 381, 407, 432]

    
    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)

        x.append(read_x_essential_features(row.data_path, select, include_bbox=include_bbox))
        #y.append(row.label)
        meta.append((row.dataset, row.augmentation,
                     row.class_name, row.file_name))
    
    pickle_load_save.save(output_file_name, (np.array(x),
                                             np.array(y).reshape((-1, 1)), np.array(meta)))
    logging.info('Dataset saved to {}'.format(output_file_name))
    
    return output_file_name
	
def create_dataset_pickle_old(csv_file=None, op_file_name='data.dat', include_bbox=False):
    """Saves a tuple of x,y and meta"""

    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data_balanced.csv')
    # if csv_file is None:
        # csv_file = path.join(Config.cs_output_root, 'all_data.csv')
    x = []
    y = []
    meta = []

    output_file_name = path.join(Config.cs_output_root, op_file_name)

    df = pd.read_csv(csv_file, index_col=0)
    #print("DF:",df['label'])

    df['data_path'] = df.apply(lambda x: get_data_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)
    df['gt_path'] = df.apply(lambda x: get_gt_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)     
    
    feature_array=[]
    bbox_array=[]
    gt_array = []
    
    
    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)
        feature_array.append(read_x_features(row.data_path, include_bbox=include_bbox))
        bbox_array.append(read_x_bbox(row.gt_path, row.class_name))
        #print("row.gt_path, bbox_array:", row.gt_path, bbox_array)
        
    feature_array=np.array(feature_array)
    select=reduction(feature_array, 3)
    
    
    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)

        x.append(read_x_essential_features(row.data_path, select, include_bbox=include_bbox))
        y.append(row.label)
        meta.append((row.dataset, row.augmentation,
                     row.class_name, row.file_name))

    pickle_load_save.save(output_file_name, (np.array(x),
                                             np.array(y).reshape((-1, 1)), np.array(meta)))
    logging.info('Dataset saved to {}'.format(output_file_name))
    
    return output_file_name

def load_dataset():
    output_file_name = path.join(Config.cs_output_root, 'data.dat')
    return pickle_load_save.load(output_file_name)

def read_x_essential_features(filename, select, include_bbox=False, all_class=False):
    
    
    class_list = Config.class_names
    all_class_names = [c.strip()
                       for c in open(Config.class_name_file).readlines()]
    if not all_class:
        class_ids = [all_class_names.index(x) for x in class_list]

    all_class_probs = []
    
    output_data = pickle_load_save.load(filename)
    
    features_raw = output_data
    features_raw=features_raw.numpy().reshape((13,13,512))
	
    # Feature reduction start
    #features_raw=features_raw[:,:,select]

    #Feature Reduction end
    #print(features_raw.shape)


    #exit()
    return features_raw


def relu(x):
    return tf.maximum(0.0, x)

def read_x_features(filename, include_bbox=False, all_class=False):
    
    
    class_list = Config.class_names
    all_class_names = [c.strip()
                       for c in open(Config.class_name_file).readlines()]
    if not all_class:
        class_ids = [all_class_names.index(x) for x in class_list]

    all_class_probs = []
    
    output_data = pickle_load_save.load(filename)
    
    features_raw = output_data
    features_raw=features_raw.numpy().reshape((13,13,512))
    
    # Apply the ReLU function to each element of the tensor
    #features_raw = tf.map_fn(relu, features_raw)
    #features_raw = (features_raw - np.min(features_raw)) / (np.max(features_raw) - np.min(features_raw))

    #exit()
    return features_raw

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def read_x_bbox(filename, class_name, all_class=False):
    
    
    class_list = Config.class_names
    
    all_class_names = [c.strip()
                       for c in open(Config.class_name_file).readlines()]
    
    if not all_class:
        class_ids = [all_class_names.index(x) for x in class_list]
    
    bboxes = []	
    cordinates = []	
    lines_list = file_lines_to_list(filename)
    true_objects = [[l.split(), 0] for l in lines_list]
    
    for true_obj in true_objects:
        bbgt = [float(x) for x in true_obj[0][1:]]
        # print("bbgt:", bbgt)
        x_px = bbgt[0] *13
        
        y_px = bbgt[1] *13
        w_px = bbgt[2] *13
        h_px = bbgt[3] *13
        xmin_px = int(x_px - (w_px/2))
        ymin_px = int(y_px - (h_px/2))
        xmax_px = int(x_px + (w_px/2))
        ymax_px = int(y_px + (h_px/2))
        xmin_px = int(max(0, xmin_px))
        ymin_px = int(max(0, ymin_px))
        xmax_px = int(min(12, xmax_px))
        ymax_px = int(min(12, ymax_px))
        pixel_cordinates = [xmin_px, xmax_px, ymin_px, ymax_px]
        cordinates.append(pixel_cordinates)
        # bboxes.append(bbgt)

    # if (xmin_px==0 or ymin_px==0):
        # print("cordinates, bbgt, x_px, w_px, xmin_px:", cordinates, bbgt, x_px, w_px, xmin_px)
        # print("Filename:", filename)
        # exit()
    return cordinates

def reduction(feature_array, y, n=2):
    
    target = y
    
    # feat_min = np.amin(feature_array,axis=(0,1,2))
    # feat_max = np.amax(feature_array,axis=(0,1,2))
    # feat_min = np.reshape(feat_min,(1, 1, 1, 512))
    # feat_max = np.reshape(feat_max,(1, 1, 1, 512))
    
    # # normalize feature_array
    # feature_array = (feature_array - feat_min) / (feat_max - feat_min)
    
    print(feature_array.shape)
    print(target)
    
    #Feature Reduction Start
    
    # Reshape the feature map to (13 * 13, 512)
    feature_array = np.sum(np.sum(feature_array, axis=1), axis=1)
    #print(feature_array[:5,:10])
    
    # Use Recursive Feature Elimination to select the top 3 features
    selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=n, step=1)
    selector = selector.fit(feature_array,target)
    
    select=[]
    # Extract only the top 100 features
    features_raw_reduced = selector.transform(feature_array)
    for i in range(n):
        for j in range(feature_array.shape[1]):
           if np.array_equal(features_raw_reduced[:,i],feature_array[:,j]):
                select.append(j) 
                break

    #Feature Reduction end
    print(select)
    
    return select
	
def reduction_new(feature_array, bbox_array, y, n=2):
    # # Initialize an empty list for the bounding box features
    bbox_feature = []
    target = y
    # feat_min = np.amin(feature_array,axis=(0,1,2))
    # feat_max = np.amax(feature_array,axis=(0,1,2))
    # feat_min = np.reshape(feat_min,(1, 1, 1, 512))
    # feat_max = np.reshape(feat_max,(1, 1, 1, 512))
    
    # # normalize feature_array
    # feature_array = (feature_array - feat_min) / (feat_max - feat_min)
    ################################################################	
    for i, bbox in enumerate(bbox_array):
        
        area_sum = np.zeros(feature_array.shape[-1])
        feature_sum = np.zeros(feature_array.shape[-1])
        for coord in bbox:
            xmin, xmax, ymin, ymax = coord
            
            feature_sum += np.sum(np.sum(feature_array[i, ymin:ymax+1, xmin:xmax+1, :], axis=0), axis=0)
            area_sum += (ymax-ymin+1)*(xmax-xmin+1) 
        bbox_feature.append(feature_sum/area_sum)
    bbox_feature = np.array(bbox_feature)
    # print("***********")
    # print(bbox_feature[:5,:10])
    # print(feature_array.shape)
    #print(xmin, xmax, ymin, ymax)
    
    #################################################################
    
    # Use RFE to select the top n features
    selector = RFE(estimator = RandomForestClassifier(), n_features_to_select=n, step=1)
    selector = selector.fit(bbox_feature, target)
    
    #selector.ranking_
    select =[]
    #Extract only the top 100 features
    features_raw_reduced = selector.transform(bbox_feature)
    for i in range(n):
        for j in range(bbox_feature.shape[1]):
           if np.array_equal(features_raw_reduced[:,i],bbox_feature[:,j]):
                select.append(j) 
                break

    #Feature Reduction end
    # print(selector.support_)
    # print(selector.ranking_)
    print(select)
    return select

   
def read_x_org(filename, include_bbox=False, all_class=False):

    class_list = Config.class_names
    all_class_names = [c.strip()
                       for c in open(Config.class_name_file).readlines()]
    if not all_class:
        class_ids = [all_class_names.index(x) for x in class_list]

    all_class_probs = []
    print("********************")
    print(filename)
    output_data = pickle_load_save.load(filename)
    bbox, objectness, class_probs = output_data

    bbox = bbox.numpy().reshape((13, 13, 12))
    objectness = objectness.numpy().reshape((13, 13, 3))

    if(all_class):
        class_probs = class_probs.numpy()
        class_probs = class_probs.reshape((13, 13, 3*80))
    else:
        for index in class_ids:
            all_class_probs.append(class_probs.numpy()[:, :, :, :, index])

        if len(class_ids) > 1:
            class_probs = np.dstack(tuple(all_class_probs))
        else:
            class_probs = all_class_probs[0]
        class_probs = class_probs.reshape((13, 13, 3*len(class_ids)))

    if(include_bbox):
        x = np.dstack((objectness, class_probs, bbox))
    else:
        x = np.dstack((objectness, class_probs))
    return x

def create_dataset():

    create_csv_of_all_detected_files()
    create_dataset_csv()
    create_dataset_pickle()


def create_dataset_taking_everything():
    pass
    

def read_file_make_dataset(list_file, output_name="dr_fp"):
    if list_file == None:
        list_file = "./data/l0check/training_dr_FP_person_b4_file_list.txt"
    file_list = file_lines_to_list(list_file)
    # Config.aug_dir_name = os.path.join('no_aug', 'person')

    field_names = ['fid', 'file_name', 'label',
                   'augmentation', 'class_name', 'dataset']

    all_data = []
    for i, filename in enumerate(file_list):
        # label_file = path.join(Config.labels_dir, filename+'.json')
        # with open(label_file) as f:
        #     label_data = json.load(f)

        data_dict = {}
        data_dict['augmentation'] = 'no_aug'
        data_dict['class_name'] = 'person'
        data_dict['dataset'] = Config.dt  # Dataset type
        data_dict['file_name'] = filename
        data_dict['label'] = 1  # label_data['label']
        all_data.append(data_dict.copy())

    csv_file = 'data/l0check/'+output_name+'.csv'
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for i, data in enumerate(all_data):
            data['fid'] = i+1
            writer.writerow(data)

    Config.aug_dir_name = ""
    output_data_file_name = create_dataset_pickle(
        csv_file, op_file_name=output_name + '.dat')

		
################################################DATASET_END#########################################################



################################################TRAIN_START#########################################################


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
    input_shape = (13, 13, 3, 2)
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

#VGG_Blocks
def make_simple_model_VGG(input_shape=(13, 13, 512)) -> keras.models.Model:
    inputs = keras.layers.Input(shape=input_shape, name='main_input')
    x = keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu')(inputs)
    #x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # x = keras.layers.Dropout(0.15)(x)
    x = keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    # x = keras.layers.Dropout(0.35)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D((3, 3), padding="same")(x)
    x = keras.layers.Dropout(0.45)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    return model

def make_simple_model(input_shape=(13, 13, 50)) -> keras.models.Model:
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

#Paper 1
def make_simple_model_paper(input_shape=(13, 13, 50)) -> keras.models.Model:
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
	

def train_bn():
    class_name = 'car'
    num = '1'
    input_data = dataset.get_detector_dataset('val', Config.class_names)
    x_train, y_train, meta_train = input_data

    # x_val, y_val, meta_val = input_data
    # x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=25)
    # y_train, y_val = train_test_split(y_train, test_size=0.15, random_state=25)
    
    ######################################
    feature_array = x_train
    target = y_train
    target = np.array(target)
    
    print(x_train.shape)
    target = target.flatten()
    print(target)
    feature_array = np.sum(np.sum(feature_array, axis=1), axis=1)
    
    n = 50
    # Use Recursive Feature Elimination to select the top 3 features
    selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=n, step=1)
    selector = selector.fit(feature_array,target)
    
    select=[]
    # Extract only the top 100 features
    features_raw_reduced = selector.transform(feature_array)
    for i in range(n):
        for j in range(feature_array.shape[1]):
           if np.array_equal(features_raw_reduced[:,i],feature_array[:,j]):
                select.append(j) 
                break

    #Feature Reduction end
    print(select)
    exit()

    ######################################
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

    #model = make_network1_alt(input_shape=(13, 13, 512))
    #model=make_model_bn()
    model = make_simple_model()
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

    model.fit(x_train, y_train, epochs=50, batch_size=50,
              validation_data=(x_val, y_val), callbacks=callbacks_list)

    # The best model has already been saved by ModelCheckpoint, so we don't need to save the model again here

    model=load_model_from_file(checkpoint_name='./checkpoints/ndt_person_cnn_bn.h5')
    test_trained_model_nn(model, model_name, x_val, y_val2=y_val)
	
	
def test_trained_model_nn(model, op_name, x_val, y_val2):

    # data_path = '/media/bijay/Projects/Datasets/val/person/outputs/data.dat'
    # x_val, y_val2, meta_val = pickle_load_save.load(data_path)
    op_dir = './data/model_results'
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)
    op_dir = os.path.join(op_dir, op_name)

    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    preds = model.predict(x_val)
    preds_05 = (preds > 0.5).astype(int)
    print("Accuracy: ")
    accuracy = accuracy_measure(y_val2, preds)
    print(accuracy)
    cm = metrics.confusion_matrix(y_val2, preds_05)
    plot_confusion_matrix_new(cm,  target_names=['False', 'True'], normalize=False,
                              title="Confusion Matrix (th={:.2f})".format(0.5), path=op_dir, name=op_name+"_cm_acc")
    TPR, FPR, th, cm = fpr_at_95_tpr(y_val2, preds)

    with open(os.path.join(op_dir, 'result.txt'), 'w') as f:
        f.write("Acc: {}\nTPR: {}\nFPR: {}\nTH: {}\n".format(
            accuracy, TPR, FPR, th))

    with open(os.path.join(op_dir, 'model.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print(TPR, "\n", FPR, "\n", th)

    plot_confusion_matrix_new(cm,  target_names=['False', 'True'], normalize=False,
                              title="Confusion Matrix (th={:.2f})".format(th), path=op_dir, name=op_name+"_cm_fpr")


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
    res = 0.001
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
  
def test_data():
    x_train, y_train, meta_train = dataset.get_detector_dataset(
        'train', Config.class_names)

    pickle_load_save.load(custom_config.detector_data_file)

    print(len(meta_train))

    # img_name = 'COCO_val2014_000000001700'
    # img_name = 'COCO_val2014_000000205636'
    img_name = 'COCO_train2014_000000000077'
    # 1,1 Baby
    # img_name = 'COCO_val2014_000000045175'  # 1,1 Skater
    # img_name = 'COCO_val2014_000000005443'  # 0,0 Skater
    # img_name = 'COCO_val2014_000000004187'  # 0,1 Nadal

    # img_name = 'COCO_val2014_000000002532'

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


################################################TRAIN_END#########################################################
if __name__ == '__main__':
    #Config.aug_dir_name = os.path.join('no_aug', 'person')
    #create_dataset()
    train_bn()