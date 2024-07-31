from tensorflow.keras.utils import plot_model
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


def run_yolo_for_tfrecord(tfrecord_filename, write_images=False):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    
    if len(physical_devices) > 0:
        for g in range(len(physical_devices)):
           tf.config.experimental.set_memory_growth(physical_devices[g], True)

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

    # make directories for storing outputs
    if not os.path.exists(Config.gt_dir):  # if it doesn't exist already
        os.makedirs(Config.gt_dir)

    if not os.path.exists(Config.gt_original_dir):  # if it doesn't exist already
        os.makedirs(Config.gt_original_dir)

    if not os.path.exists(Config.dr_dir):  # if it doesn't exist already
        os.makedirs(Config.dr_dir)

    if not os.path.exists(Config.dr_original_dir):  # if it doesn't exist already
        os.makedirs(Config.dr_original_dir)

    if not os.path.exists(Config.img_optional_dir):  # if it doesn't exist already
        os.makedirs(Config.img_optional_dir)

    if not os.path.exists(Config.output_data_dir):
        os.makedirs(Config.output_data_dir)

    network_data = []

    for img_raw, label, fname in tqdm.tqdm(iters, desc='Running Detections', unit='Images'):

        orig_filename = fname.numpy().decode('utf-8')
        filename = orig_filename.replace('.jpg', ".txt")
        data_filename = orig_filename.replace('.jpg', '.data')
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, Config.size)
        boxes_final, boxes_raw = yolo(img)

        boxes, scores, classes, nums = boxes_final

        # get_detection_box_info(boxes_raw, boxes, nums[0])
        # detections1 = print_row_col_bbox(boxes_raw[0][0], boxes, nums[0])
        # print(detections1)
        # network_data.append(boxes_raw[0])

        classes = np.array(classes[0])
        boxes = np.array(boxes[0])
        scores = np.array(scores[0])
        ground_truth_lines = []
        detection_lines = []

        wh = np.flip(img_raw.shape[0:2])

        for valid_label in label:
            if(sum(valid_label) != 0.0):
                x1y1 = tuple(
                    (np.array(valid_label[0:2]) * wh).astype(np.int32))
                x2y2 = tuple(
                    (np.array(valid_label[2:4]) * wh).astype(np.int32))

                stri = "{0} {1} {2} {3} {4} {5}\n".format(
                    class_names[int(valid_label[4])].replace(' ', ''), valid_label[0], valid_label[1], valid_label[2], valid_label[3], valid_label[5])

                # stri = "{0} {1} {2} {3} {4}\n".format(
                #     class_names[int(valid_label[4])].replace(' ', ''), x1y1[0], x1y1[1], x2y2[0], x2y2[1])

                ground_truth_lines.append(stri)

        for i in range(nums[0]):
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

            stri = "{0} {1} {2} {3} {4} {5}\n".format(
                class_names[int(classes[i])].replace(' ', ''), scores[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

            # stri = "{0} {1} {2} {3} {4} {5}\n".format(
            #     class_names[int(classes[i])].replace(' ', ''), scores[i], x1y1[0], x1y1[1], x2y2[0], x2y2[1])

            detection_lines.append(stri)

        with open(os.path.join(Config.gt_original_dir, filename), 'w') as fp:
            fp.writelines(ground_truth_lines)

        with open(os.path.join(Config.dr_original_dir, filename), 'w') as fp:
            fp.writelines(detection_lines)

        pickle_load_save.save(os.path.join(
            Config.output_data_dir, data_filename), boxes_raw[0])

        if write_images:
            img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(
                Config.img_optional_dir, orig_filename), img)

        total_records += 1
        # print(total_records)

    logging.info('output saved to: {}'.format(Config.dr_original_dir))


def print_row_col_bbox(outputs, bboxes, num):
    detections = []
    for out1 in outputs:
        for row, grid_row in enumerate(out1):
            for col, grid_col in enumerate(grid_row):
                for b, bbox in enumerate(grid_col):
                    for i in range(num):
                        eq_vec = bbox.numpy() == bboxes[0][i].numpy()
                        if(np.count_nonzero(eq_vec) >= 3):
                            # if((eq_vec).all()):
                            detections.append((row, col, b))

    return detections


def main(args):
    run_yolo_for_tfrecord(os.path.join(
        Config.tfrecord_dir, f'{Config.cs_dir_name}_annotation.tfrecord'))


if __name__ == '__main__':
    app.run(main)
