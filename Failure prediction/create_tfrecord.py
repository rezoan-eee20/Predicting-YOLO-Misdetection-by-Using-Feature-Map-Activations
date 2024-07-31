import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image
import pycocotools

import glob

import tensorflow as tf
import tools.tf_record_creation_util as tf_record_creation_util
from tools import label_map_util
from absl import app, flags, logging
from absl.flags import FLAGS
from config import Config
import logging


def test_category_index(_):
    annotations_file = FLAGS.val_annotations_file

    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.io.gfile.GFile(annotations_file, 'r') as fid:
        groundtruth_data = json.load(fid)
        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        for i in range(1, 81):
            print(i, ':', category_index[i]['name'])


def create_tf_example(image, annotations_list, image_dir, category_index, include_masks):
    """COnvert image and annotations to a tf.Example proto
    Args:
        image: dict with keys:
        [u'License', u'filename', u'coco_url', u'height',
            u'width', u'date_captured', u'flickr_url',u'id']
        annotations_list:
        list of dicts with keys:
        ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
        Notice that the bounding box coordinates in the official coco dataset are given as [x,y,width,height]
        tuples with absolute coordinates where x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
      image_dir: directory containing the image files
      category_index: a dict containing COCO category information keyed by the id field of each catagory. See the
      label_map_util.create_category_index function.

    Returns:
        example: The converted tf.Example
        num_annotations_skipped: NUmber of invalid annotations that were ignored

    Raises:
        ValueError: if the image pointed to by data[filename] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    class_names = [c.strip() for c in open(Config.class_name_file).readlines()]

    full_path = os.path.join(image_dir, filename)
    
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0

    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if width > image_width or height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x)/image_width)
        xmax.append(float(x+width)/image_width)
        ymin.append(float(y)/image_height)
        ymax.append(float(y+height)/image_height)

        is_crowd.append(object_annotations['iscrowd'])

        category_id = int(object_annotations['category_id'])

        category_id_trans = class_names.index(
            category_index[category_id]['name'])

        # category_ids.append(category_id)
        category_ids.append(category_id_trans)

        category_names.append(
            category_index[category_id]['name'].encode('utf8'))

        area.append(object_annotations['area'])

    feature_dict = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_id).encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=category_names)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=category_ids)),
        'image/object/is_crowd': tf.train.Feature(int64_list=tf.train.Int64List(value=is_crowd)),
        # 'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        # 'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }

    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def create_tf_record(annotations_file, image_dir, output_path, include_masks, num_shards, image_list_file):
    """Loads COCO annotation json files and converts to tf.Record format.
    Args:
      annotations_file: JSON file containing bounding box annotations.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
      num_shards: number of output file shards.
    """

    """Only includes the files in the image_list_file to output in output"""

    if(image_list_file):
        with open(image_list_file, 'r') as file_list:
            valid_images = [line.strip() for line in file_list.readlines()]

    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.io.gfile.GFile(annotations_file, 'r') as fid:

        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)

        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']

        if(image_list_file):
            print(images[0])
            images = [im for im in images if im['file_name'] in valid_images]

        print("HEYYYYY")
        print(len(images))

        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            logging.info(
                'Found groundtruth annotations. Building annotations index.')
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)
        missing_annotation_count = 0

        for image in images:

            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        logging.info('%d images are missing annotations.',
                     missing_annotation_count)

        total_num_annotations_skipped = 0

        writer = tf.io.TFRecordWriter(output_path)

        for idx, image in enumerate(images):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(images))
            annotations_list = annotations_index[image['id']]
            _, tf_example, num_annotations_skipped = create_tf_example(
                image, annotations_list, image_dir, category_index, include_masks)
            total_num_annotations_skipped += num_annotations_skipped
            shard_idx = idx % num_shards
            # output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            writer.write(tf_example.SerializeToString())

        logging.info('Finished writing, skipped %d annotations.',
                     total_num_annotations_skipped)


def create_tf_record_from_annotation(images_dir, annotation_base_name='', annotation_path=''):
    if not os.path.exists(Config.tfrecord_dir):
        os.makedirs(Config.tfrecord_dir)

    if(annotation_path != ''):
        af = annotation_path
        annotation_base_name = os.path.basename(annotation_path).split('.')[0]
    else:
        af = os.path.join(Config.annotation_dir,
                          annotation_base_name + '_annotations.json')

    output_file = os.path.join(
        Config.tfrecord_dir, annotation_base_name+'.tfrecord')
    print(output_file)
   
    create_tf_record(af, images_dir,
                     output_file, False, num_shards=1, image_list_file=None)


def create_tf_record_for_files_in_annotation_dir(class_names):
    if not os.path.exists(Config.tfrecord_dir):
        os.makedirs(Config.tfrecord_dir)

    annotation_directory = Config.annotation_dir
    ann_files = glob.glob(annotation_directory + '/*.json')

    for af in ann_files:
        if(af.find('_aug_') > 0):
            continue

        filename = os.path.split(af)[1].split('.')[0].split('_')[0]
        logging.debug('Making tfrecord file from {}'.format(af))
        output_file = os.path.join(Config.tfrecord_dir, filename+'.tfrecord')

        create_tf_record(af, Config.root_images_dir,
                         output_file, False, num_shards=1, image_list_file=None)


if __name__ == '__main__':
    # pass
    #     # app.run(test_category_index)
    #     app.run(main)
    # create_tf_record_for_specific_class(['car', 'person'])
    print(Config.root_images_dir)
    create_tf_record_from_annotation(
        Config.root_images_dir, annotation_path=os.path.join(Config.dataset_dir, Config.annotation_dir_name, f'{Config.cs_dir_name}_annotation.json'))

