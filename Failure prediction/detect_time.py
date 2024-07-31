import time

import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

# import plot_results

import pickle
import pickle_load_save
import tqdm

from config import Config
import json
import logging

    
def run_yolo_tfrecord_time(tfrecord_filename, write_images=False):
    # # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    
    # # if len(physical_devices) > 0:
        # # for g in range(len(physical_devices)):
           # # tf.config.experimental.set_memory_growth(physical_devices[g], True)

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    ngpus = max(strategy.num_replicas_in_sync,1)
    print("Number of devices: {}".format(ngpus))

    with strategy.scope():
        if Config.tiny:
            yolo = YoloV3Tiny(classes=Config.num_classes)
        else:
            yolo = YoloV3(classes=Config.num_classes)
    
    yolo.load_weights(Config.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(Config.class_name_file).readlines()]
    logging.info('classes loaded')

    dataset = load_tfrecord_dataset(
        tfrecord_filename, Config.classes, Config.size, with_filename=True)

    print('Loaded tfrecord, running detections')

    iters = iter(dataset)

    total_records = 0
    image_list = []
    for img_raw, label, fname in tqdm.tqdm(iters, desc='Running Detections', unit='Images'):
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, Config.size)
        img = img.numpy()
        image_list.append(img)
        total_records += 1
        if total_records == 8192:
            break

    image_list = np.squeeze(np.array(image_list))
    image_list = tf.data.Dataset.from_tensor_slices(image_list)
    # The batch size must now be set on the Dataset objects.
    image_list = image_list.batch(32*ngpus)
    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    image_list = image_list.with_options(options)

    start = time.time()
    yolo.predict(image_list)
    end = time.time()

    inference_time = end - start
    image_inference_time = inference_time/total_records
    with open(os.path.join('./data/', 'YOLO_Inference_Time.txt'), 'w') as f:
        f.write("Inference_time: {}\nPer_image_inference_time: {}\n".format(inference_time, image_inference_time))


def main(args):
    print('TFRECORD FILE NAME:')
    print(Config.tfrecord_dir, f'{Config.cs_dir_name}_annotation.tfrecord')
    run_yolo_tfrecord_time(os.path.join(
        Config.tfrecord_dir, f'{Config.cs_dir_name}_annotation.tfrecord'))


if __name__ == '__main__':
    app.run(main)
