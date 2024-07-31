from colorama import Fore, Back, Style
import numpy
import pickle_load_save
from config import Config
import os
import detector_new.draw_image as di
import detector_new.dataset as dataset
from detector_new.create_labels import file_lines_to_list, save_file_list
import pprint
import json
import glob
import sys
from tqdm import tqdm
import numpy as np


Config.aug_dir_name = os.path.join('no_aug', 'person')


def error(msg):
    print(msg)
    sys.exit(0)


def get_is_crowd_flag():
    x, y, meta = dataset.get_detector_dataset('val')
    crowds = [0] * len(meta)
    for i, m in enumerate(meta):
        gts = file_lines_to_list(os.path.join(
            Config.gt_original_dir, m[3]+'.txt'))
        for line in gts:
            class_name, left, top, right, bottom, iscrowd = line.split()
            # print(iscrowd, m[3])
            if(iscrowd == '1.0'):
                crowds[i] = 1
                continue

    print(sum(crowds))
    return crowds


def get_data():
    x, y, meta = dataset.get_detector_dataset('train')
    if not (os.path.exists('test_images')):
        os.makedirs('test_images')

    print(y[0:10])
    for i, m in enumerate(meta):
        if(y[i] == 1):
            di.draw_img_test_file(m[3], 'test_images', Config.gt_dir)

    print(len(x))
    print(meta[0])


def make_proper_label(fileid):
    classname = 'person'
    label_dir = Config.labels_dir

    gt_orig = file_lines_to_list(os.path.join(
        Config.gt_original_dir, fileid+'.txt'))

    dr_orig = file_lines_to_list(os.path.join(
        Config.dr_original_dir, fileid+'.txt'))

    # [[line], is_detected, index_where_match,iou_value, area]

    true_objects = [[l.split(), 0, -1, 0.0, 0.0]
                    for l in gt_orig if l.split()[0] == classname]
    pred_objects = [[l.split(), 0, -1, 0.0, 0.0]
                    for l in dr_orig if l.split()[0] == classname]
    matches = 0

    for pi, pred_obj in enumerate(pred_objects):
        bb = [float(x) for x in pred_obj[0][2:]]

        for ti, true_obj in enumerate(true_objects):
            if(true_obj[1] == 1):
                continue
            ov = 0.0
            maxiou = 0

            if pred_obj[0][0] == true_obj[0][0]:  # same class
                bbgt = [float(x) for x in true_obj[0][1:]]
                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(
                    bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                      + 1) * (bbgt[3] - bbgt[1] + 1) - (iw * ih)
                    ov = (iw * ih) / ua

                # print(ov)
                if(ov > true_obj[3]):
                    true_obj[3] = ov

                if(ov > pred_obj[3]):
                    pred_obj[3] = ov

                if(ov >= Config.threshold):
                    matches += 1

        for ti, true_obj in enumerate(true_objects):
            if true_obj[1] == 1:
                continue

            bbgt = [float(x) for x in true_obj[0][1:]]
            true_obj[4] = (bbgt[2] - bbgt[0]) * (bbgt[3] - bbgt[1])

            if true_obj[3] == pred_obj[3] and true_obj[3] >= Config.threshold:
                true_obj[1] = 1
                pred_obj[1] = 1
                true_obj[2] = pi
                pred_obj[2] = ti
            else:
                true_obj[1] = 0
                true_obj[3] = 0.0

    true_objects_filtered = [
        to for to in true_objects if (to[4] > Config.area_threshold or to[1] == 1)]

    true_count = len(true_objects_filtered)
    pred_count = len(pred_objects)

    # print(Fore.RED)
    # pp = pprint.PrettyPrinter(indent=1)
    # # pp.pprint(true_objects)

    # for ttto in true_objects:
    #     print(ttto)

    # print(Fore.GREEN)

    # # pp.pprint(pred_objects)

    # for ppo in pred_objects:
    #     print(ppo)

    # print(Fore.YELLOW)
    # for ppo in true_objects_filtered:
    #     print(ppo)

    # print(Style.RESET_ALL)

    # number detected in ground truth array
    tos = sum([x[1]
               for x in true_objects_filtered if x[3] >= Config.threshold])

    # number detected in pred array
    pos = sum([x[1] for x in pred_objects if x[3] >= Config.threshold])
    # 1 or 0 in label
    is_prefect = (true_count == pred_count) and (
        tos == pos) and (true_count == pos)

    detected = [x[0] for x in pred_objects if x[1] == 1]
    extra_detections = [x[0] for x in pred_objects if x[1] == 0]
    missed = [x[0] for x in true_objects_filtered if x[1] == 0]

    label = {"label": int(is_prefect),
             "filename": fileid,
             "gt_count": true_count,
             "dr_count": pred_count,
             "dr_match_count": tos,
             "gt_match_count": pos,
             "detected_list": detected,
             "false_positive": extra_detections,
             "missed": missed,
             "gt_per_obj_label": true_objects}

    # print(label)
    # print(true_count)
    # print(len(true_objects_filtered))
    # print(pred_count)

    # print(is_prefect)

    with open(os.path.join(Config.labels_dir, fileid+".json"), 'w') as f:
        json.dump(label, f)

    if not os.path.exists(Config.gt_dir):
        os.makedirs(Config.gt_dir)
    with open(os.path.join(Config.gt_dir, fileid+'.txt'), 'w') as gt_of:
        for tof in true_objects_filtered:
            gt_of.write(' '. join(tof[0]))
            gt_of.write('\n')

    # di.draw_img_test_file(fileid, 'test_images', Config.gt_original_dir)

    return label


def find_IOU(pred_obj, true_obj):
    bb = [float(x) for x in pred_obj[0][2:]]
    bbgt = [float(x) for x in true_obj[0][1:]]

    bbgt = [float(x) for x in true_obj[0][1:]]
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(
        bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1

    ov = 0.0

    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - (iw * ih)
        ov = (iw * ih) / ua

    print(ov)


def check_if_list_has_crowd(fileid):
    linelist = file_lines_to_list(os.path.join(
        Config.gt_original_dir, fileid+'.txt'))
    for line in linelist:
        class_name, left, top, right, bottom, iscrowd = line.split()
        if(iscrowd == '1.0'):
            return True

    return False


def make_label_files_for_all():
    label_dir = Config.labels_dir
    gt_dir = Config.gt_original_dir

    label_0_files = []
    label_1_files_with_detections = []
    label_1_files_without_detections = []
    label_0_missed = []
    label_0_false_positives = []

    all_labels = []
    crowd_files = []

    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    ground_truth_files_list = glob.glob(gt_dir + '/*.txt')

    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()

    for txt_file in tqdm(ground_truth_files_list, 'Creating labels'):
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)  # gt file contents

        if check_if_list_has_crowd(file_id) == True:
            crowd_files.append(file_id)
            continue

        label = make_proper_label(file_id)

        all_labels.append(label)
        filename_for_list = file_id+'.jpg'

        if(label['label'] == 1):
            if(label['gt_count'] == 0 and label['dr_count'] == 0):
                label_1_files_without_detections.append(filename_for_list)
            else:
                label_1_files_with_detections.append(filename_for_list)

        else:
            label_0_files.append(filename_for_list)
            label_0_files.append(filename_for_list)

            if(len(label['missed']) > 0):
                label_0_missed.append(filename_for_list)

            if(len(label['false_positive']) > 0):
                label_0_false_positives.append(filename_for_list)

    save_file_list('l0', label_0_files)
    save_file_list('l1', label_1_files_with_detections +
                   label_1_files_without_detections)
    save_file_list('l1_without_detections', label_1_files_without_detections)
    save_file_list('l1_with_detections', label_1_files_with_detections)
    save_file_list('l0_missed', label_0_missed)
    save_file_list('l0_fp', label_0_false_positives)

    save_file_list('is_crowd', crowd_files)

    with open(os.path.join(Config.cs_output_root, "all_labels.json"), 'w') as f:
        json.dump(all_labels, f)


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


if __name__ == "__main__":
    # get_is_crowd_flag()
    # get_data()
    make_label_files_for_all()

    # make_proper_label('COCO_val2014_000000000395')
    # make_proper_label('COCO_train2014_000000549127')
    # make_proper_label('COCO_train2014_000000026483')
    # make_proper_label('COCO_val2014_000000000761')
    # ps = 'person 0.8988845348358154 0.24343743920326233 0.5415383577346802 0.7083109617233276 0.9601349830627441'

    # ts = 'person 0.1938125044107437 0.8288333415985107 0.4702187478542328 0.9883958101272583 0.0'
    # # ts = 'person 0.057218749076128006 0.6870416402816772 0.22703124582767487 0.9974166750907898 0.0'

    # po = [ps.split()]
    # to = [ts.split()]

    # find_IOU(po, to)
    # bb = [float(x) for x in po[0][2:]]
    # bbgt = [float(x) for x in to[0][1:]]
    # print(get_iou(bbgt, bb))
    # exit(0)

    di.draw_img_gt_dr('COCO_train2014_000000018885',
                      gt='/media/bijay/Projects/Datasets/train/person/outputs/no_aug/person/gt.txt',
                      dr='/media/bijay/Projects/Datasets/train/person/outputs/no_aug/person/dt.txt',
                      output_dir='/media/bijay/Projects/Datasets/train/person/outputs/no_aug/person')

    test_files = [
        'COCO_train2014_000000000308',
        'COCO_train2014_000000018885',
        'COCO_train2014_000000001942',
        'COCO_train2014_000000005288',
        'COCO_train2014_000000007500',
        'COCO_train2014_000000007794',
        'COCO_train2014_000000007583',
        'COCO_train2014_000000009025',
        'COCO_train2014_000000009057',
        'COCO_train2014_000000009744',
        'COCO_train2014_000000011075',
        'COCO_train2014_000000011292',
        'COCO_train2014_000000012044'
    ]
    if not (os.path.exists('test_images')):
        os.makedirs('test_images')

    for f in test_files[1:2]:
        lbl = make_proper_label(f)
        print(lbl['label'])
