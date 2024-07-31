import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from absl import flags, app, logging
from absl.flags import FLAGS
import glob
import json
import pickle_load_save
import io
import hashlib
from datetime import datetime
from fnmatch import fnmatch
import random
import math

AUTO = tf.data.experimental.AUTOTUNE  # used in tf.data.Dataset API

filepath = "./data/network_op.pickle"
flags.DEFINE_string(
    'LABELS_dir', '/media/bijay/Projects/Datasets/outputs/labels', 'directory with labels')
flags.DEFINE_string('OUTPUT_files', '/media/bijay/Projects/Datasets/outputs/data',
                    'dataset image directory. Used for animation')

flags.DEFINE_string('TRAIN_tfrecord', './data/detector_train.tfrecord',
                    'training tfrecord file to create')
flags.DEFINE_string('TEST_tfrecord', './data/detector_val.tfrecord',
                    'training tfrecord file to create')
flags.DEFINE_string('VAL_tfrecord', './data/detector_val.tfrecord',
                    'training tfrecord file to create')


# DATASET_SIZE = 40400
DATASET_SIZE = 100
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.05 * DATASET_SIZE)
test_size = int(0.25 * DATASET_SIZE)


def get_training_validation_test_filelist(data_dir):

    file_list = [file for file in os.listdir(
        data_dir) if fnmatch(file, '*.data')]

    random.shuffle(file_list)
    train_split = 0.7
    val_split = 0.1
    split_index = math.floor(len(file_list) * train_split)
    training = file_list[:split_index]
    testing = file_list[split_index:]

    val_split_index = math.floor(len(training) * val_split)

    validation = training[val_split_index:]
    training = training[:val_split_index]
    return training, validation, testing


def get_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_all_labels(_):
    labels_file_list = glob.glob(FLAGS.LABELS_dir + '/*.json')

    print(len(labels_file_list))
    if(len(labels_file_list) == 0):
        print("No label files found")

    labels = []
    for label_json in labels_file_list:
        jo = get_json(label_json)
        labels.append(jo['label'])
    print('DOne appending')
    print(len([x for x in labels if x == 0]))
    print(labels[:10])
    return labels


def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray"""
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:
        raise ValueError(
            "The input should be numpy ndarray. Instead got {}".format(ndarray.dtype))


def create_tf_example(data_file_path, label_file_path):
    output_data = pickle_load_save.load(data_file_path)
    bbox, objectness, class_probs = output_data
    bbox = bbox.numpy().reshape((13, 13, 12))
    objectness = objectness.numpy().reshape((13, 13, 3))
    class_probs = class_probs.numpy().reshape((13, 13, 240))
    x = np.dstack((bbox, objectness, class_probs))
    x = x.reshape((13*13*255))

    dtype_x = _dtype_feature(x)

    assert isinstance(x, np.ndarray)

    # output_data_to_io = io.BytesIO(alldata)
    # key = hashlib.sha256(alldata).hexdigest()

    y = [get_json(label_file_path)['label']]
    y = np.array(y)
    dtype_y = _dtype_feature(y)

    y_sh = np.asarray(y.shape)
    dtype_ysh = _dtype_feature(y_sh)

    x_sh = np.asarray(x.shape)
    dtype_xsh = _dtype_feature(x_sh)

    feature_dict = {
        'X': dtype_x(x),
        'x_shape': dtype_xsh(x_sh),
        'Y': dtype_y(y),
        'y_shape': dtype_ysh(y_sh)
    }

    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))

    return 0, example


def create_tfrecord(data_file_list, output_file):
    # data_file_list = glob.glob(FLAGS.OUTPUT_files + '/*.data')
    with tf.io.TFRecordWriter(output_file) as writer:
        for data_file in data_file_list:
            file_id = data_file.split(".data", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))

            label_path = os.path.join(FLAGS.LABELS_dir, file_id+'.json')
            _, example = create_tf_example(data_file, label_path)
            writer.write(example.SerializeToString())

    print("Finished writing file")


def make_tf_records():
    train_filelist, val_filelist, test_filelist = get_training_validation_test_filelist(
        FLAGS.OUTPUT_files)

    logging.debug('Creating trainging tfrecord')
    create_tfrecord(train_filelist, FLAGS.TRAIN_tfrecord)

    logging.debug('Creating validation tfrecord')
    create_tfrecord(val_filelist, FLAGS.VAL_tfrecord)

    logging.debug('Creating testing tfrecord')
    create_tfrecord(test_filelist, FLAGS.TEST_tfrecord)


def parse_tf_record(example):

    features = {
        "X": tf.io.VarLenFeature(tf.float32),
        "Y": tf.io.VarLenFeature(tf.int64),
        "x_shape": tf.io.VarLenFeature(tf.int64),
        "y_shape": tf.io.VarLenFeature(tf.int64)
    }
    # features = {
    #     "X": tf.io.FixedLenFeature([], tf.float32),
    #     "Y": tf.io.FixedLenFeature([], tf.int64),

    #     # "x_shape": tf.io.FixedLenFeature([], tf.int64),
    #     # "y_shape": tf.io.FixedLenFeature([], tf.int64)
    # }

    example = tf.io.parse_single_example(example, features)

    x = example["X"]
    # x = tf.cast(x, tf.float32)
    x = tf.sparse.reshape(x, (13, 13, 255))
    x = tf.sparse.to_dense(x)
    y = example["Y"]
    y = tf.sparse.reshape(y, (1, 1))
    y = tf.sparse.to_dense(y)

    # X = tf.expand_dims(x, axis=-1)

    # x_sh = example["x_shape"]
    # y_sh = example["y_shape"]

    return x, y


def get_tf_record(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tf_record)
    return dataset.take(20000)


def unpickle_load_data(filepath):
    with open(filepath, 'rb') as rf:
        data = pickle.load(rf)
    print(len(data))
    data_final = []
    for dt in data:
        bbox, objectness, class_probs = dt
        bbox = bbox.numpy().reshape((13, 13, 12))
        objectness = objectness.numpy().reshape((13, 13, 3))
        class_probs = class_probs.numpy().reshape((13, 13, 240))
        alldata = np.dstack((bbox, objectness, class_probs))
        data_final.append(alldata)
        # print(alldata.shape)

    return np.array(data_final)


def make_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(13, 13, 255)))
    # model.add(keras.layers.Dense(43095, activation='relu'))
    # model.add(keras.layers.Dense(20000, activation='relu'))
    model.add(keras.layers.Dense(1000, activation='relu'))
    model.add(keras.layers.Dense(600, activation='relu'))
    model.add(keras.layers.Dense(80, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    opt_adam = keras.optimizers.Adam()

    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def get_training_test_dataset():
    tf.random.set_seed(42)
    dataset = get_tf_record(output_file)
    dataset.shuffle(buffer_size=40000)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    # drop_remainder will be needed on TPU

    train_dataset = dataset.batch(64, drop_remainder=True)
    test_dataset = dataset.batch(64, drop_remainder=True)
    val_dataset = val_dataset.batch(64, drop_remainder=True)

    return train_dataset, val_dataset, test_dataset


def train(_):
    model = make_model()
    training_set, val_set, test_set = get_training_test_dataset()
    filepath = "./checkpoints/new/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    tensorboard = keras.callbacks.TensorBoard(log_dir='logs/detector_{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")
                                                                                ))
    callbacks_list = [checkpoint]
    model.fit(training_set, epochs=100, callbacks=[tensorboard])
    #   validation_data=val_set,
    #   callbacks=callbacks_list)

    model.save('./checkpoints/my_checkmodel_2k.h5')


def load_model_and_test(_):
    model = tf.keras.models.load_model('./checkpoints/my_checkmodel_2k.h5')
    training_set, val_set, test_set = get_training_test_dataset()
    model.evaluate(test_set)


def main(_):
    training, validation, testing = separate_val_train_test_tfrecord()
    print(len(training))
    print(len(validation))
    print(len(testing))


if __name__ == "__main__":
    # app.run(create_tfrecord)
    # app.run(train)
    # app.run(load_model_and_test)
    app.run(main)
