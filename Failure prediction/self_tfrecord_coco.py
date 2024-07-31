import tensorflow as tf 
import json

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}

raw_dataset = tf.data.TFRecordDataset("Person.tfrecord")
size = 416
class_file = './data/coco.names'

LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)



for raw_record in raw_dataset:
    example = tf.train.Example()

    x = tf.io.parse_single_example(raw_record, IMAGE_FEATURE_MAP)

    # example.ParseFromString(raw_record.numpy())

    #  m = json.loads(MessageToJson(ex))
    # print(m['features']['feature'].keys())
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))
    filename = x['image/filename']
    isCrowd = x['image/object/is_crowd']

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')

    labels = tf.cast(class_table.lookup(class_text), tf.float32)

    isCrowd = tf.cast(isCrowd, tf.float32)
    
    a= tf.sparse.to_dense(x['image/object/bbox/xmin'])
    # print(a)
    b= tf.sparse.to_dense(x['image/object/bbox/ymin'])
    # print(a)

    c= tf.sparse.to_dense(x['image/object/bbox/xmax'])
    # print(a)

    d= tf.sparse.to_dense(x['image/object/bbox/ymax'])
    # print(a)

    is_crowd_sparse = tf.sparse.to_dense(isCrowd)

    y_train = tf.stack([a,b,c,d, labels], axis=1)

    # y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
    #                     tf.sparse.to_dense(x['image/object/bbox/ymin']),
    #                     tf.sparse.to_dense(x['image/object/bbox/xmax']),
    #                     tf.sparse.to_dense(x['image/object/bbox/ymax']),
    #                     labels,
    #                     isCrowd], axis=1)

    # paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]

    # y_train = tf.pad(y_train, paddings)

    # print(y_train)

