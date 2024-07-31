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
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
import random
import tensorflow as tf
import detector_new.dataset as dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

    field_names = ['fid', 'file_name', 'label', 'augmentation', 'class_name', 'dataset']

    all_class_name_combinations = Config.get_all_class_names_combinations()

    output_dir_subfolder = [f.name for f in os.scandir(Config.output_dir) if f.is_dir()]
    sub_folder = Config.output_dir
    for cls_name in all_class_name_combinations:
        Config.cs_dir_name = os.path.join(sub_folder, cls_name)
        if not os.path.exists(Config.cs_output_root):
            logging.warn('Not found: Ignoring: {}'.format(Config.cs_dir_name))
            continue
        
        logging.info('found: processing: {}'.format(
            Config.cs_dir_name))

        label_1_source = path.join(Config.cs_file_list_dir, 'l1_with_detections.txt')
        
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
        print("Config.cs_file_list_dir:", Config.cs_file_list_dir)
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
                Config.cs_file_list_dir, 'l1_with_detections_person.txt')
            label_1_source = path.join(
                Config.cs_file_list_dir, 'l1_with_detections_car.txt')

            label_2_source = path.join(
                Config.cs_file_list_dir, 'l0_person.txt')
            label_3_source = path.join(
                Config.cs_file_list_dir, 'l0_car.txt')

            #label_4_source = path.join(Config.cs_file_list_dir, 'l0_fp.txt')

            # label_0_source = path.join(Config.cs_file_list_dir, 'l0.txt')
            label0 = file_lines_to_list(label_0_source)
            label1 = file_lines_to_list(label_1_source)
            label2 = file_lines_to_list(label_2_source)
            label3 = file_lines_to_list(label_3_source)
            #label4 = file_lines_to_list(label_4_source)

            label_array = [label0, label1, label2, label3]

            lines += len(label1) + len(label0) + \
                len(label2) + len(label3)

            data_dict = {}

            if cls_name not in stat:
                stat[cls_name] = {0: len(label0), 1: len(label1), 2: len(
                    label2), 3: len(label3)}
            else:
                stat[cls_name][0] += len(label0)
                stat[cls_name][1] += len(label1)
                stat[cls_name][2] += len(label2)
                stat[cls_name][3] += len(label3)
                #stat[cls_name][4] += len(label4)

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
    print("all_class_names:", all_class_names[0:2])
    data = pd.read_csv(csv_file, index_col=0)
    all_data_coll = pd.DataFrame(columns=data.columns)

    for cls_name in all_class_names[0:2]:
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
            [all_data_coll,  label1, label0, label2, label3, label4])

    all_data_coll.to_csv(output_csv)
    logging.info('Saved to {}'.format(output_csv))


def get_data_path(aug, class_name, file_name):
    file_name=str(file_name)
    return os.path.join(Config.cs_output_root, Config._output_data_dir_name, file_name+ '.data')

def get_gt_path(aug, class_name, file_name):
    file_name=str(file_name)
    return os.path.join(Config.cs_output_root, Config._gt_dir_name, file_name+ '.txt')

def create_dataset_pickle_paper(csv_file=None, op_file_name='data.dat', include_bbox=False):
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


def create_dataset_pickle_new(csv_file=None, op_file_name='data.dat', include_bbox=False):
    """Saves a tuple of x,y and meta"""

    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data.csv')
    # if csv_file is None:
        # csv_file = path.join(Config.cs_output_root, 'all_data.csv')
    x = []
    y = []
    meta = []

    output_file_name = path.join(Config.cs_output_root, op_file_name)

    df = pd.read_csv(csv_file, index_col=0)
    

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
    y = np.array(y).reshape((-1, 1))
    bbox_feature_avg_plot(feature_array, bbox_array, y)
    
    #select=reduction(feature_array, y, 84)
    #select=reduction(feature_array, bbox_array, y, 25)
    #select = [2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,25,26,27,28]
    #select = [1, 5, 14, 59, 62, 70, 71, 73, 74, 76, 103, 107, 110, 121, 126, 127, 130, 137, 141, 152, 173, 189, 195, 199, 220, 235, 243, 247, 257, 283, 291, 308, 313, 327, 332, 334, 366, 370, 380, 381, 387, 389, 404, 407, 462, 482, 488, 493, 507, 511]
    #select_car_25 = [5, 14, 59, 62, 73, 74, 76, 107, 121, 126, 127, 139, 199, 220, 247, 257, 283, 291, 308, 332, 381, 387, 404, 407, 432]
    #select = [14, 59, 73, 74, 76, 126, 127, 220, 247, 257, 308, 332, 381, 407, 432]
    #RFC50 [1, 5, 14, 51, 59, 62, 70, 71, 73, 74, 76, 103, 107, 110, 117, 121, 126, 127, 137, 151, 152, 173, 189, 199, 220, 225, 235, 243, 247, 257, 283, 291, 308, 327, 331, 332, 334, 341, 366, 380, 381, 387, 389, 404, 407, 431, 432, 488, 493, 511]

    
    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)

        x.append(read_x_essential_features(row.data_path, select, include_bbox=include_bbox))
        #y.append(row.label)
        meta.append((row.dataset, row.augmentation,
                     row.class_name, row.file_name))
    
    pickle_load_save.save(output_file_name, (np.array(x),
                                             y, np.array(meta)))
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
    #select=reduction(feature_array, 3)
    select = []
    
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

def create_dataset_pickle(csv_file=None, op_file_name='data.dat', include_bbox=False, ):
    """Saves a tuple of x,y and meta"""

    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data_balanced.csv')

    output_file_name = path.join(Config.cs_output_root, op_file_name)

    df = pd.read_csv(csv_file, index_col=0)
    

    df['data_path'] = df.apply(lambda x: get_data_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)
    df['gt_path'] = df.apply(lambda x: get_gt_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)     

    x, y, meta = [], [], []
    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)
        y.append(row.label)
        x.append(read_x_features(row.data_path, include_bbox=include_bbox))
        meta.append((row.dataset, row.augmentation,
                     row.class_name, row.file_name)+tuple([int(getattr(row, x)) for x in row._fields if ('label' in x or '_class' in x)]))

    x = np.array(x)
    y = np.array(y).reshape((-1, 1))
    meta = np.array(meta)

    """ Remove features using recursive feature elimination """
    # # select = reduction(x, y, 84)
    # # x, meta = [], []
    # # for row in tqdm.tqdm(df.itertuples()):
        # # x.append(read_x_essential_features(row.data_path, select, include_bbox=include_bbox))
        # # meta.append((row.dataset, row.augmentation,
                     # # row.class_name, row.file_name))
    # # x = np.array(x)
    # # meta = np.array(meta)

    pickle_load_save.save(output_file_name, (x, y, meta))
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
	
    #Feature reduction start
    features_raw=features_raw[:,:,select]

    return features_raw

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
        pixel_cordinates = [xmin_px, xmax_px, ymin_px, ymax_px,class_ids]
        cordinates.append(pixel_cordinates)
        # bboxes.append(bbgt)

    # if (xmin_px==0 or ymin_px==0):
        # print("cordinates, bbgt, x_px, w_px, xmin_px:", cordinates, bbgt, x_px, w_px, xmin_px)
        # print("Filename:", filename)
        # exit()
    return cordinates

def reduction(feature_array, y, n=2):
    
    target = y.flatten()
    x_train = feature_array
    y_train = target	
    x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, y_val = train_test_split(y_train, test_size=0.15, random_state=25)
    
    feat_min = np.amin(x_train,axis=(0,1,2))
    feat_max = np.amax(x_train,axis=(0,1,2))
    feat_min = np.reshape(feat_min,(1, 1, 1, 512))
    feat_max = np.reshape(feat_max,(1, 1, 1, 512))
    
    # normalize feature_array
    x_train = (x_train - feat_min) / (feat_max - feat_min)
    
    print(x_train.shape, y_train.shape)
    
    #Feature Reduction Start
    
    # Reshape the feature map to (13 * 13, 512)
    x_train = np.sum(np.sum(x_train, axis=1), axis=1)
    #print(feature_array[:5,:10])
    
    # Use Recursive Feature Elimination to select the top 3 features
    selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=n, step=5)
    selector = selector.fit(x_train,y_train)
    # selector = RFECV(estimator=DecisionTreeClassifier(), step=1, cv = 5)
    # selector = selector.fit(feature_array,target)
    # rf = RandomForestClassifier(random_state=42)
    # selector = RFECV(estimator=rf, cv=StratifiedKFold(5), scoring='accuracy')
    # selector = selector.fit(feature_array,target)
    select = np.argwhere(selector.support_)[:,0].tolist()

    #Feature Reduction end
    print(select)
  
    return select

def reduction_cnn(feature_array, y, n=2):
    
    y_train = target = y.flatten()
    print(feature_array.shape, target.shape)
    feature_array = np.sum(np.sum(feature_array, axis=1), axis=1)
    X_train = feature_array
    # Create CNN model
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=10, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create RFE object
    selector = RFE(estimator=model, n_features_to_select=50, step=1)
    selector = selector.fit(feature_array,target)
    select = np.argwhere(selector.support_)[:,0].tolist()

    #Feature Reduction end
    print(select)
    
    return select
	
def reduction_bbox_org(feature_array, bbox_array, y, n=2):
    # # Initialize an empty list for the bounding box features
    bbox_feature = []
    target = y.flatten()
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
    select = np.argwhere(selector.support_)[:,0].tolist()
    
    print(select)
    return select

def reduction_bbox(feature_array, bbox_array, y, n=2):
    # # Initialize an empty list for the bounding box features
    target = y.flatten()
    bbox_feature = []
    x_train = feature_array
    y_train = target	
    x_train, x_val = train_test_split(x_train, test_size=0.15, random_state=25)
    y_train, y_val = train_test_split(y_train, test_size=0.15, random_state=25)
    # target = y
    feat_min = np.amin(x_train,axis=(0,1,2))
    feat_max = np.amax(x_train,axis=(0,1,2))
    feat_min = np.reshape(feat_min,(1, 1, 1, 512))
    feat_max = np.reshape(feat_max,(1, 1, 1, 512))
    
    # normalize feature_array
    x_train = (x_train - feat_min) / (feat_max - feat_min)
    ################################################################

    for i, bbox in enumerate(bbox_array):
        
        area_sum = np.zeros(x_train.shape[-1])
        feature_sum = np.zeros(x_train.shape[-1])
        for coord in bbox:
            xmin, xmax, ymin, ymax = coord
            
            feature_sum += np.sum(np.sum(x_train[i, ymin:ymax+1, xmin:xmax+1, :], axis=0), axis=0)
            area_sum += (ymax-ymin+1)*(xmax-xmin+1) 
        bbox_feature.append(feature_sum/area_sum)
    bbox_feature = np.array(bbox_feature)
    # print("***********")
    # print(bbox_feature[:5,:10])
    # print(feature_array.shape)
    #print(xmin, xmax, ymin, ymax)
    
    #################################################################
    
    # Use RFE to select the top n features
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # selector = RFECV(estimator=rf, cv=StratifiedKFold(5), scoring='accuracy')
    selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=n, step=5)
    selector = selector.fit(bbox_feature,target)
    select = np.argwhere(selector.support_)[:,0].tolist()
    
    print(select)
    return select 

def bbox_feature_avg_func_old(class_name, csv_file=None, op_file_name='data.dat', include_bbox=False):
    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data_balanced.csv')
    x = []
    y = []
    
    output_file_name = path.join(Config.cs_output_root, op_file_name)

    df = pd.read_csv(csv_file, index_col=0)
    

    df['data_path'] = df.apply(lambda x: get_data_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)
    df['gt_path'] = df.apply(lambda x: get_gt_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)     
    
    feature_array = []
    bbox_array = []    
    
    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)
        y.append(row.label)
        feature_array.append(read_x_features(row.data_path, include_bbox=include_bbox))
        bbox_array.append(read_x_bbox(row.gt_path, row.class_name))
        
    bbox_array = np.array(bbox_array) 
    feature_array = np.array(feature_array)
    bbox_feature = []
    x_train = feature_array	
    
    for i, bbox in enumerate(bbox_array):
        
        area_sum = 0
        feature_sum = np.zeros(x_train.shape[-1])
        for coord in bbox:
            xmin, xmax, ymin, ymax, class_ids = coord
            
            feature_sum += np.sum(np.sum(x_train[i, ymin:ymax+1, xmin:xmax+1, :], axis=0), axis=0)
            area_sum += (ymax-ymin+1)*(xmax-xmin+1) 
			
        if len(bbox)!=0: bbox_feature.append(feature_sum/area_sum)
    bbox_feature = np.array(bbox_feature)
    bbox_feature_avg =  np.average(bbox_feature, axis=0).flatten()
	
    return bbox_feature_avg, bbox_feature
    

def bbox_feature_avg_func(class_name, csv_file=None, op_file_name='data.dat', include_bbox=False):
    if csv_file is None:
        csv_file = path.join(Config.cs_output_root, 'all_data_balanced.csv')
    x = []
    y_0 = []
    y_1 = []
    

    output_file_name = path.join(Config.cs_output_root, op_file_name)

    df = pd.read_csv(csv_file, index_col=0)
    

    df['data_path'] = df.apply(lambda x: get_data_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)
    df['gt_path'] = df.apply(lambda x: get_gt_path(
        x['augmentation'], x['class_name'], x['file_name']), 1)     
    
    feature_array_0 = []
    feature_array_1 = []
    bbox_array_0 = []
    bbox_array_1 = []    
    
    
    for row in tqdm.tqdm(df.itertuples()):
        # print('----', row.data_path)
        #y.append(row.label)
        if row.label ==0:
           y_0.append(row.label)
           feature_array_0.append(read_x_features(row.data_path, include_bbox=include_bbox))
           bbox_array_0.append(read_x_bbox(row.gt_path, row.class_name))
        else:
           y_1.append(row.label)
           feature_array_1.append(read_x_features(row.data_path, include_bbox=include_bbox))
           bbox_array_1.append(read_x_bbox(row.gt_path, row.class_name))
    bbox_array_0 = np.array(bbox_array_0) 
    bbox_array_1 = np.array(bbox_array_1)
    feature_array_0 = np.array(feature_array_0)
    feature_array_1 = np.array(feature_array_1)
  	
    bbox_feature_0 = []
    bbox_feature_1 = []
    x_train_0 = feature_array_0
    x_train_1 = feature_array_1	
    
    for i, bbox in enumerate(bbox_array_0):
        
        area_sum_0 = 0
        feature_sum_0 = np.zeros(x_train_0.shape[-1])
        for coord in bbox:
            xmin, xmax, ymin, ymax, class_ids = coord
            
            feature_sum_0 += np.sum(np.sum(x_train_0[i, ymin:ymax+1, xmin:xmax+1, :], axis=0), axis=0)
            area_sum_0 += (ymax-ymin+1)*(xmax-xmin+1) 
			
        if len(bbox)!=0: bbox_feature_0.append(feature_sum_0/area_sum_0)
    bbox_feature_0 = np.array(bbox_feature_0)
    bbox_feature_avg_0 =  np.average(bbox_feature_0, axis=0).flatten()
	
    for i, bbox in enumerate(bbox_array_1):
        
        area_sum_1 = 0
        feature_sum_1 = np.zeros(x_train_1.shape[-1])
        for coord in bbox:
            xmin, xmax, ymin, ymax, class_ids = coord
            
            feature_sum_1 += np.sum(np.sum(x_train_1[i, ymin:ymax+1, xmin:xmax+1, :], axis=0), axis=0)
            area_sum_1 += (ymax-ymin+1)*(xmax-xmin+1) 
			
        if len(bbox)!=0: bbox_feature_1.append(feature_sum_1/area_sum_1)
    bbox_feature_1 = np.array(bbox_feature_1)
    bbox_feature_avg_1 =  np.average(bbox_feature_1, axis=0).flatten()
	
    return bbox_feature_avg_0, bbox_feature_0, bbox_feature_avg_1, bbox_feature_1
	
def bbox_feature_avg_plot_clss():
    Config.class_names=['person']
    Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    box_feature_avg_person_failure, bbox_feature_person_failure, bbox_feature_avg_person_success, bbox_feature_person_success = bbox_feature_avg_func(class_name = 'person')
    X = np.arange(1, len(box_feature_avg_person_failure)+1, 1)
    bbox_feature_person = np.vstack([bbox_feature_person_failure, bbox_feature_person_success])
    
    Config.class_names=['car']
    Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    bbox_feature_avg_car_failure, bbox_feature_car_failure, bbox_feature_avg_car_success, bbox_feature_car_success = bbox_feature_avg_func(class_name = 'car')
    #X = np.arange(1, len(bbox_feature_avg_car)+1, 1)
    bbox_feature_car = np.vstack([bbox_feature_car_failure, bbox_feature_car_success])
    
    Config.class_names=['cup']
    Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    bbox_feature_avg_cup_failure, bbox_feature_cup_failure, bbox_feature_avg_cup_success, bbox_feature_cup_success = bbox_feature_avg_func(class_name = 'cup')
    #X = np.arange(1, len(bbox_feature_avg_cup)+1, 1)
    bbox_feature_cup = np.vstack([bbox_feature_cup_failure, bbox_feature_cup_success])
    
    Config.class_names=['chair']
    Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    box_feature_avg_chair_failure, bbox_feature_chair_failure, bbox_feature_avg_chair_success, bbox_feature_chair_success = bbox_feature_avg_func(class_name = 'chair')
    #X = np.arange(1, len(bbox_feature_avg_chair)+1, 1)
    bbox_feature_chair = np.vstack([bbox_feature_chair_failure, bbox_feature_chair_success])
	
	# Example class labels
    class_labels = ['person', 'car', 'cup', 'chair']
    # Define colors for each class label
    colors = ['green', 'red', 'blue', 'yellow']
    
    pca = PCA(n_components=4)
    pca.fit(np.vstack([bbox_feature_person, bbox_feature_car, bbox_feature_cup, bbox_feature_chair]))
    pca_person = pca.transform(bbox_feature_person)
    pca_car = pca.transform(bbox_feature_car)
    pca_cup = pca.transform(bbox_feature_cup)
    pca_chair = pca.transform(bbox_feature_chair)
    # Create scatter plot
    plt.scatter(pca_person[:, 0], pca_person[:, 1], color= colors[0], label= class_labels[0])
    plt.scatter(pca_car[:, 0], pca_car[:, 1], color= colors[1], label= class_labels[1])
    plt.scatter(pca_cup[:, 0], pca_cup[:, 1], color= colors[2], label= class_labels[2])
    plt.scatter(pca_chair[:, 0], pca_chair[:, 1], color= colors[3], label= class_labels[3])
    
    plt.xlabel('PCA_1')
    plt.ylabel('PCA_2')
    plt.legend()
    
    # Show the plot
    plt.show()
    exit()
    # # # plt.plot(Xp, bbox_feature_avg_p, label = "person")
    # # # plt.plot(Xcar, bbox_feature_avg_car, label = "car")
    # # plt.plot(Xcup, bbox_feature_avg_cup, label = "cup")
    # # # plt.plot(Xchair, bbox_feature_avg_chair, label = "chair")
    
    # # plt.legend()
    # # plt.show()
    
    # # # bbox_feature_avg_p[::-1].sort()
    # # # plt.plot(Xp, bbox_feature_avg_p, label = "person")
	
    # # # bbox_feature_avg_car[::-1].sort()
    # # # plt.plot(Xcar, bbox_feature_avg_car, label = "car")
	
    # # bbox_feature_avg_cup[::-1].sort()
    # # plt.plot(Xcup, bbox_feature_avg_cup, label = "cup")
	
    # # # bbox_feature_avg_chair[::-1].sort()
    # # # plt.plot(Xchair, bbox_feature_avg_chair, label = "chair")
    
    plt.legend()
    plt.show()


def bbox_feature_avg_plot_failure():
    # # Config.class_names=['person']
    # # Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    # # bbox_feature_avg_p_0,failure_feature_maps, bbox_feature_avg_p_1, success_feature_maps = bbox_feature_avg_func(class_name = 'person')
    # # X = np.arange(1, len(bbox_feature_avg_p_0)+1, 1)
    Config.class_names=['car']
    Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    bbox_feature_avg_car_0,failure_feature_maps, bbox_feature_avg_car_1, success_feature_maps = bbox_feature_avg_func(class_name = 'car')
    X = np.arange(1, len(bbox_feature_avg_car_0)+1, 1)
    
    # # Config.class_names=['bottle']
    # # Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    # # bbox_feature_avg_bottle_0,failure_feature_maps, bbox_feature_avg_bottle_1, success_feature_maps = bbox_feature_avg_func(class_name = 'bottle')
    # # X = np.arange(1, len(bbox_feature_avg_bottle_0)+1, 1)
	
    # Config.class_names=['chair']
    # Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    # bbox_feature_avg_chair_0, failure_feature_maps, bbox_feature_avg_chair_1, success_feature_maps  = bbox_feature_avg_func(class_name = 'chair')
    # X = np.arange(1, len(bbox_feature_avg_chair_0)+1, 1)
    # Config.class_names=['cup']
    # Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    # bbox_feature_avg_cup_0, failure_feature_maps, bbox_feature_avg_cup_1, success_feature_maps = bbox_feature_avg_func(class_name = 'cup')
    # X = np.arange(1, len(bbox_feature_avg_cup_0)+1, 1)    
    
    # # Assuming you have your feature maps stored in two arrays: success_feature_maps and failure_feature_maps
    # print("success_feature_maps:", success_feature_maps.shape)
    # # Compute cosine similarity matrix
    # cosine_similarity_matrix = cosine_similarity(success_feature_maps, failure_feature_maps)
    #print(cosine_similarity_matrix)
    
    # Example class labels
    class_labels = ['Success', 'Failure']
    # Define colors for each class label
    colors = ['green', 'red']
    # Compute PCA to reduce the similarity matrix to 2 dimensions
    pca = PCA(n_components=2)
    # reduced_similarity = pca.fit_transform(cosine_similarity_matrix)
    pca.fit(np.vstack([failure_feature_maps, success_feature_maps]))
    
    pca_failure = pca.transform(failure_feature_maps)
    pca_success = pca.transform(success_feature_maps)
    # Create scatter plot
    # for i in range(len(reduced_similarity)):
        # plt.scatter(reduced_similarity[i, 0], reduced_similarity[i, 1], color=colors[i % len(colors)])
    
	# Create scatter plot
    plt.scatter(pca_failure[:, 0], pca_failure[:, 1], color= colors[0], label= "Failure")
    plt.scatter(pca_success[:, 0], pca_success[:, 1], color= colors[1], label= "Success")
    
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    # # Add class labels to the scatter plot
    # for i, label in enumerate(class_labels):
        # plt.annotate(label,(reduced_similarity[i, 0], reduced_similarity[i, 1]))
    
    # Show the plot
    plt.show()
    exit()
	
	
    # feature_avg = np.vstack([bbox_feature_avg_cup_0, bbox_feature_avg_cup_1])
    # print(feature_avg.shape) 
    
	# # Compute cosine similarity matrix
    # cos_sim_matrix = cosine_similarity(feature_avg)
	
    # # Reduce dimensionality with t-SNE
    # tsne = TSNE(n_components=2, random_state=42)
    # reduced_features = tsne.fit_transform(cos_sim_matrix)
    
    # # Visualize the reduced features
    # plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    # plt.title("Visualization of High-Dimensional Features")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.show()
	
    # classes = ["label_0", "label_1"]
    # features  = [str(i+1) for i in range(512)]
    
    # feature_avg = np.vstack([bbox_feature_avg_cup_0, bbox_feature_avg_cup_1])
    # print(feature_avg.shape) 
    
	# # Compute cosine similarity matrix
    # # cos_sim_matrix = cosine_similarity(feature_avg)
    # # fig, ax = plt.subplots()
    # # im = ax.imshow(cos_sim_matrix, cmap='hot', interpolation='nearest')	
    
    # fig, ax = plt.subplots()
    # im = ax.imshow(feature_avg)
    
    # # Show all ticks and label them with the respective list entries
    # #ax.set_xticks(np.arange(len(features)), labels=features)
    # ax.set_yticks(np.arange(len(classes)), labels=classes)
    
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             # rotation_mode="anchor")
    
    # # Loop over data dimensions and create text annotations.
    # # for i in range(len(classes)):
        # # for j in range(len(features)):
            # # text = ax.text(j, i, feature_avg[i, j],
                           # # ha="center", va="center", color="w")
    
    # ax.set_title("Heatmap")
    # #fig.tight_layout()
    # #zoom_factor = 100 
	# # Set the zoomed-in range
    # ax.set_xlim(0, 50)  # Adjust the range according to your desired zoom level
    # ax.set_ylim(0, 1)  # Adjust the range according to your desired zoom level
    # plt.show()	

    
    
    
    
   
    # Config.class_names=['chair']
    # Config.cs_dir_name = Config.get_cs_dir_name(Config.class_names)
    # bbox_feature_avg_chair_0, bbox_feature_avg_chair_1  = bbox_feature_avg_func(class_name = 'chair')
    # X = np.arange(1, len(bbox_feature_avg_chair_0)+1, 1)
    
	
    # # plt.plot(X, np.absolute(bbox_feature_avg_cup_0 - bbox_feature_avg_cup_1))
    
    # plt.plot(X, bbox_feature_avg_p_0, label = "person_0")
    # plt.plot(X, bbox_feature_avg_p_1, label = "person_1")
    # plt.plot(X, bbox_feature_avg_car_0, label = "car_0")
    # plt.plot(X, bbox_feature_avg_car_1, label = "car_1")
    # plt.plot(X, bbox_feature_avg_cup_0, label = "cup_0")
    # plt.plot(X, bbox_feature_avg_cup_1, label = "cup_1")
    # plt.plot(X, bbox_feature_avg_chair_0, label = "chair_0")
    # plt.plot(X, bbox_feature_avg_chair_1, label = "chair_1")
    
    plt.legend()
    plt.show()
    
    bbox_feature_avg_p_0[::-1].sort()
    plt.plot(X, bbox_feature_avg_p_0, label = "person_0")
    bbox_feature_avg_p_1[::-1].sort()
    plt.plot(X, bbox_feature_avg_p_1, label = "person_1")
	
    # bbox_feature_avg_car_0[::-1].sort()
    # plt.plot(X, bbox_feature_avg_car_0, label = "car_0")
    # bbox_feature_avg_car_1[::-1].sort()
    # plt.plot(X, bbox_feature_avg_car_1, label = "car_1")
	
    # bbox_feature_avg_cup_0[::-1].sort()
    # plt.plot(X, bbox_feature_avg_cup_0, label = "cup_0")
    # bbox_feature_avg_cup_1[::-1].sort()
    # plt.plot(X, bbox_feature_avg_cup_1, label = "cup_1")
    
    # bbox_feature_avg_chair_0[::-1].sort()
    # plt.plot(X, bbox_feature_avg_chair_0, label = "chair_0")
    # bbox_feature_avg_chair_1[::-1].sort()
    # plt.plot(X, bbox_feature_avg_chair_1, label = "chair_1")
    
    plt.legend()
    plt.show()	
      
	
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

    # create_csv_of_all_detected_files()
    # create_dataset_csv()
    create_dataset_pickle()
    #bbox_feature_avg_plot_failure()
    #bbox_feature_avg_plot_clss()

    # all_data_csv = path.join(Config.cs_output_root, 'all_data.csv')
    # create_dataset_pickle(all_data_csv, op_file_name='unbalanced_data.dat')


def create_dataset_taking_everything():
    pass
    # create_dataset()
    # x, y, meta = load_dataset()

    # print(x.shape, y.shape, meta.shape)
    # get_x_y('/media/bijay/Projects/Datasets/val/car_truck/outputs/no_aug/car/data/COCO_val2014_000000073731.data')


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

    # x, y, meta = pickle_load_save.load(output_data_file_name)
    # print(len(x))
    # print(len(x[0]))


if __name__ == '__main__':
    #Config.aug_dir_name = os.path.join('no_aug', 'person')
    create_dataset()
    # create_dataset_pickle(op_file_name='with_bbox.data')

    # read_file_make_dataset(
    #     './data/nlstats/dt_minus_crowd_train.txt', 'new_dt_train.dat')

    # read_file_make_dataset(
    #     './data/l0check/validation_dr_TP_one_person_b4_file_list.txt', 'validation_dr_TP_one_person_b4_file_list.dat')
    # Config.aug_dir_name = os.path.join('no_aug', 'person')

    # Config.aug_dir_name = os.path.join('no_aug', 'mixed')
    # read_file_make_dataset(
    #     './data/l0check/truenegative.txt', 'true_negative.dat')

    # Config = ConfigDTS('val', ['person'])
    # Config.aug_dir_name = os.path.join('no_aug', 'person')
    # read_file_make_dataset(
    #     './data/l0check/validation_dr_FP_person_b4_file_list.txt', 'validation_dr_FP_person_b4.dat')
    # Config.aug_dir_name = os.path.join('no_aug', 'person')

    # read_file_make_dataset(
    #     './data/l0check/validation_gt_FN_person_b4_file_list.txt', 'validation_gt_FN_person_b4.dat')

    # Config.aug_dir_name = ""
    # x, y, meta = pickle_load_save.load(
    #     path.join(Config.cs_output_root, 'validation_dr_FP_person_b4.dat.dat'))
    # print(x.shape)
    # print(meta)
