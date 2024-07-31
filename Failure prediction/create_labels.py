from absl import flags, app
import os
import glob
import sys
import pickle_load_save
from config import Config
import json
import logging
from tqdm import tqdm
import detector_new.draw_image as di
import random
import pprint


def error(msg):
    print(msg)
    sys.exit(0)


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def filter_DR(class_names):
    """Filter the detection result to keep only classes in class names"""
    
    
    detection_result_files_list = glob.glob(Config.dr_original_dir + '/*.txt')
    if len(detection_result_files_list) == 0:
        error("Error: No ground-truth files found! at " + Config.dr_original_dir)
    detection_result_files_list.sort()
    
    for txt_file in tqdm(detection_result_files_list, 'Filtering DR'):
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)
        new_dr_lines = []
        
        for line in lines_list:
            class_name, _, _, _, _, _ = line.split()
            bb = [float(x) for x in line.split()[2:]]
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            
            # # if(class_name in class_names and area > Config.area_threshold):
                # # new_dr_lines.append(line + "\n")
            if(class_name in class_names):
                new_dr_lines.append(line + "\n")
        
        
        with open(os.path.join(Config.dr_dir, file_id+".txt"), 'w') as ofile:
            ofile.writelines(new_dr_lines)

def filter_GT(class_names):
    """Filter the detection result to keep only classes in class names"""
    
    ground_truth_files_list = glob.glob(Config.gt_original_dir + '/*.txt')
    print("Config.gt_original_dir", Config.gt_original_dir)
    
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found! at " + Config.gt_original_dir)
    ground_truth_files_list.sort()
    for txt_file in tqdm(ground_truth_files_list, 'Filtering GT'):
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)
        new_gt_lines = []
        
        for line in lines_list:
            class_name, _, _, _, _, _ = line.split()
            bb = [float(x) for x in line.split()[2:]]
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            
            # # if(class_name in class_names and area > Config.area_threshold):
                # # new_gt_lines.append(line + "\n")
            if(class_name in class_names):
                new_gt_lines.append(line + "\n")
                
                 
        
        with open(os.path.join(Config.gt_dir, file_id+".txt"), 'w') as ofile:
            ofile.writelines(new_gt_lines)
    
    

def filter_GT_single_file(class_names, input_file, output_dir):
    file_id = input_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    lines_list = file_lines_to_list(input_file)
    pred_file = os.path.join(Config.dr_dir, file_id+".txt")
    pred_lines_list = file_lines_to_list(pred_file)
    pred_objects = [[l.split(), 0, -1, 0.0] for l in pred_lines_list]
    new_gt_lines = []
    for line in lines_list:
        class_name, left, top, right, bottom, iscrowd = line.split()
        width = float(right) - float(left)
        height = float(bottom) - float(top)

        area = width * height

        if(class_name in class_names):
            if(area > Config.area_threshold and iscrowd != '1.0'):
                new_gt_lines.append(line + "\n")
            else:
                pred_objects, is_T = check_is_in_detection_extra(
                    file_id, line, pred_objects)

                if(is_T == True and iscrowd != '1.0'):
                    new_gt_lines.append(line + "\n")

    with open(os.path.join(output_dir, file_id+".txt"), 'w') as ofile:
        ofile.writelines(new_gt_lines)


def filter_GT_Org(class_names):
    """Filters ground truth based on AREA"""

    # get a list with the ground-truth files
    # if not os.path.isdir(FLAGS.GT_files_filtered):
    #     os.makedirs(FLAGS.GT_files_filtered)

    ground_truth_files_list = glob.glob(Config.gt_original_dir + '/*.txt')
    
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in tqdm(ground_truth_files_list, 'Filtering GT'):
        # print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)
        new_gt_lines = []

        pred_file = os.path.join(Config.dr_dir, file_id+".txt")
        pred_lines_list = file_lines_to_list(pred_file)
        pred_objects = [[l.split(), 0, -1, 0.0] for l in pred_lines_list]

        for line in lines_list:
            class_name, left, top, right, bottom, iscrowd = line.split()
            width = float(right) - float(left)
            height = float(bottom) - float(top)

            area = width * height

            if(class_name in class_names):
                 if(area > Config.area_threshold and iscrowd != '1.0'):
                     new_gt_lines.append(line + "\n")
                 else:
                     pred_objects, is_T = check_is_in_detection_extra(
                         file_id, line, pred_objects)

                     if(is_T == True and iscrowd != '1.0'):
                         new_gt_lines.append(line + "\n")
            
        with open(os.path.join(Config.gt_dir, file_id+".txt"), 'w') as ofile:
            ofile.writelines(new_gt_lines)

    # os.rename(FLAGS.GT_files, FLAGS.GT_files + "_original")
    # os.rename(FLAGS.GT_files_filtered, FLAGS.GT_files)


def save_file_list(filename, image_name_list):
    output_dir = Config.cs_file_list_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, filename+'.txt'), 'w') as opf:
        opf.write('\n'.join(image_name_list))


def check_if_list_has_crowd(fileid):
    linelist = file_lines_to_list(os.path.join(
        Config.gt_original_dir, fileid+'.txt'))
    for line in linelist:
        class_name, left, top, right, bottom, iscrowd = line.split()
        # print(iscrowd, m[3])
        if(iscrowd == '1.0'):
            return True

    return False


def quick_check():
    lcrowd = file_lines_to_list(os.path.join(
        Config.cs_file_list_dir, 'is_crowd'+'.txt'))
    scrowd = set(lcrowd)
    l1 = file_lines_to_list(os.path.join(Config.cs_file_list_dir, 'l1'+'.txt'))
    l0 = file_lines_to_list(os.path.join(Config.cs_file_list_dir, 'l0'+'.txt'))
    s0 = set(l0)
    s1 = set(l1)

    ins1 = scrowd.intersection(l0)
    ins2 = scrowd.intersection(l1)

    print(len(ins1), len(ins2))


def make_detector_labels_extra(separate_root=None):
    """Creates a label file"""
    label_dir = Config.labels_dir
    gt_dir = Config.gt_dir
    dr_dir = Config.dr_dir
    label_0_files = []
    label_1_files_with_detections = []
    label_1_files_without_detections = []
    label_0_missed = []
    label_0_false_positives = []

    all_labels = []
    crowd_files = []
    
    if(separate_root != None):
        label_dir = os.path.join(separate_root, 'labels')
        gt_dir = os.path.join(separate_root, 'ground-truth')
        dr_dir = os.path.join(separate_root, 'detection-results')
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)

    ground_truth_files_list = glob.glob(gt_dir + '/*.txt')
    detection_files_list = glob.glob(dr_dir + '/*.txt')
    
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    detection_files_list.sort()

    for txt_file in tqdm(ground_truth_files_list, 'Creating labels'):

        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        lines_list = file_lines_to_list(txt_file)  # gt file contents

        if check_if_list_has_crowd(file_id) == True:
            crowd_files.append(file_id)
            continue

        pred_file = os.path.join(dr_dir, file_id+".txt")
        pred_lines_list = file_lines_to_list(pred_file)  # dr file contents

        true_objects = [[l.split(), 0, -1, 0.0] for l in lines_list]
        pred_objects = [[l.split(), 0, -1, 0.0] for l in pred_lines_list]

        match_indices = []
        matches = 0

        for pi, pred_obj in enumerate(pred_objects):
            # if pred_obj[1] == 1:
            #     continue
            bb = [float(x) for x in pred_obj[0][2:]]

            for ti, true_obj in enumerate(true_objects):

                # the prediction already matches with another gt
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
                        # trread_file_make_datasetue_obj[3] = ov

                        # true_obj[1] = 1
                        # pred_obj[1] = 1
                        # true_obj[2] = pi
                        # pred_obj[2] = ti
                        # break
                else:
                    print("Label did not match")

            for ti, true_obj in enumerate(true_objects):
                if true_obj[1] == 1:
                    continue

                if true_obj[3] == pred_obj[3] and true_obj[3] >= Config.threshold:
                    true_obj[1] = 1
                    pred_obj[1] = 1
                    true_obj[2] = pi
                    pred_obj[2] = ti
                else:
                    true_obj[1] = 0
                    true_obj[3] = 0.0

        true_count = len(true_objects)
        pred_count = len(pred_objects)
        # number detected in ground truth array
        tos = sum([x[1] for x in true_objects])
        # number detected in pred array
        pos = sum([x[1] for x in pred_objects])
        # 1 or 0 in label
        is_prefect = (true_count == pred_count) and (
            tos == pos) and (true_count == pos)
        detected = [x[0] for x in pred_objects if x[1] == 1]
        extra_detections = [x[0] for x in pred_objects if x[1] == 0]
        missed = [x[0] for x in true_objects if x[1] == 0]
        # print(true_objects)
        # write all info to label file so that it can be used later if needed
        label = {"label": int(is_prefect),
                 "filename": file_id,
                 "gt_count": true_count,
                 "dr_count": pred_count,
                 "dr_match_count": tos,
                 "gt_match_count": pos,
                 "detected_list": detected,
                 "false_positive": extra_detections,
                 "missed": missed,
                 "gt_per_obj_label": true_objects}
        all_labels.append(label)

        with open(os.path.join(label_dir, file_id+".json"), 'w') as f:
            json.dump(label, f)

        filename_for_list = file_id+'.jpg'
        if(is_prefect):
            if true_count == 0 and pred_count == 0:
                label_1_files_without_detections.append(filename_for_list)
            else:
                label_1_files_with_detections.append(filename_for_list)
        else:
            label_0_files.append(filename_for_list)

            if(len(missed) > 0):
                label_0_missed.append(filename_for_list)

            if(len(extra_detections) > 0):
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


def check_is_in_detection_extra(file_id, line, pred_objects):
    # pred_file = os.path.join(Config.dr_dir, file_id+".txt")
    # pred_lines_list = file_lines_to_list(pred_file)
    true_objects = [[l.split(), 0, -1, 0.0] for l in [line]]
    # pred_objects = [[l.split(), 0, -1, 0.0] for l in pred_lines_list]
    match_indices = []
    matches = 0

    # print(true_objects)

    for pi, pred_obj in enumerate(pred_objects):
        if pred_obj[1] == 1:
            continue
        bb = [float(x) for x in pred_obj[0][2:]]
        for ti, true_obj in enumerate(true_objects):
            # the prediction already matches with another gt
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

                area = ((bi[2] - bi[0])*(bi[3] - bi[1]))
                # print(area)
                # print(area < Config.area_threshold)

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
                    # true_obj[3] = ov

                    # true_obj[1] = 1
                    # pred_obj[1] = 1
                    # true_obj[2] = pi
                    # pred_obj[2] = ti
                    # break
            else:
                print("Label did not match")

        for ti, true_obj in enumerate(true_objects):
            if true_obj[1] == 1:
                continue

            if true_obj[3] == pred_obj[3] and true_obj[3] >= Config.threshold:

                true_obj[1] = 1
                pred_obj[1] = 1
                true_obj[2] = pi
                pred_obj[2] = ti
            else:
                true_obj[1] = 0
                true_obj[3] = 0.0

    if true_objects[0][1] == 1:
        # # print("Matched_True:", true_objects)
        # # print("Matched_True:", true_objects[0][1])
        # # print("Matched_Pred:", pred_objects)
        # # exit()
        return pred_objects, True
    # # print("Unmatched_True:", true_objects[0][1])
    # # print("Unmatched_Pred:", pred_objects)
    
    return pred_objects, False
    # print("Matches ", matches)
    # pp = pprint.PrettyPrinter(depth=5, compact=True)
    # pp.pprint(true_objects)
    # print('-------------\n')
    # pp.pprint(pred_objects)

    # true_count = len(true_objects)
    # pred_count = len(pred_objects)
    # # number detected in ground truth array
    # tos = sum([x[1] for x in true_objects])
    # # number detected in pred array
    # pos = sum([x[1] for x in pred_objects])
    # # 1 or 0 in label
    # is_prefect = (true_count == pred_count) and (
    #     tos == pos) and (true_count == pos)
    # detected = [x[0] for x in pred_objects if x[1] == 1]
    # extra_detections = [x[0] for x in pred_objects if x[1] == 0]
    # missed = [x[0] for x in true_objects if x[1] == 0]

    # # print(true_objects)
    # # write all info to label file so that it can be used later if needed
    # label = {"label": int(is_prefect),
    #          "filename": 'file_id',
    #          "gt_count": true_count,
    #          "dr_count": pred_count,
    #          "dr_match_count": tos,
    #          "gt_match_count": pos,
    #          "detected_list": detected,
    #          "false_positive": extra_detections,
    #          "missed": missed,
    #          "gt_per_obj_label": true_objects}

    # return label


def check_is_in_detection(file_id, line):

    pred_file = os.path.join(Config.dr_dir, file_id+".txt")
    pred_lines_list = file_lines_to_list(pred_file)

    true_split = line.split(' ')
    # print(true_split)

    for pl in pred_lines_list:
        pred_split = pl.split(' ')
        if(pred_split[0] == true_split[0]):  # If it matches
            bbgt = [float(x) for x in true_split[1:]]
            bb = [float(x) for x in pred_split[2:]]

            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(
                bb[2], bbgt[2]), min(bb[3], bbgt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            if iw > 0 and ih > 0:
                # compute overlap (IoU) = area of intersection / area of union
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                  + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                ov = iw * ih / ua
                # if ov > ovmax:
                #     ovmax = ov
                #     gt_match = obj
            if(ov >= Config.threshold):
                return True
                # print('detected')

    return False

	
def make_detector_labels(separate_root=r'/home/local2/Ferdous/YOLO/Datasets/val/chair/outputs'):
    """Creates a label file"""
    label_dir = Config.labels_dir
    gt_dir = Config.gt_dir
    dr_dir = Config.dr_dir

    label_0_files = []

    label_1_files_with_detections = []
    label_1_files_without_detections = []

    label_0_missed_some = []
    label_0_missed_all = []

    label_0_false_positives = []
    label_0_missed_extra = []

    label_0_no_missed_no_extra = []  # This should be empty if everything is allright

    all_labels = []

    if(separate_root != None):
        label_dir = os.path.join(separate_root, 'labels')
        gt_dir = os.path.join(separate_root, 'ground-truth')
        dr_dir = os.path.join(separate_root, 'detection-results')
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)

    ground_truth_files_list = glob.glob(gt_dir + '/*.txt')
    detection_files_list = glob.glob(dr_dir + '/*.txt')

    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    detection_files_list.sort()

    for txt_file in tqdm(ground_truth_files_list, 'Creating labels'):
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)  # gt file contents
        pred_file = os.path.join(dr_dir, file_id+".txt")
        pred_lines_list = file_lines_to_list(pred_file)  # dr file contents

        true_objects = [[l.split(), 0] for l in lines_list]
        pred_objects = [[l.split(), 0] for l in pred_lines_list]
        match_indices = []

        # # count1=count2=0
        for pred_obj in pred_objects:
            bb = [float(x) for x in pred_obj[0][2:]]
            # # count1+=1 
            # # count2=0
            for true_obj in true_objects:
                # # count2+=1
                # # # the prediction already matches with another gt
                # # if(true_obj[1] == 1):
                    # # continue
                if pred_obj[0][0] == true_obj[0][0]:  # same class
                    bbgt = [float(x) for x in true_obj[0][1:]]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(
                        bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                    # # print(count1,count2,ov)
                    if(ov >= Config.threshold):
                        true_obj[1] = 1
                        pred_obj[1] = 1
                        # # break

        match_flag = 1
        miss_flag = 0
        pred_area_flag = 0
        true_area_flag = 0		
		
        for pred_obj in pred_objects:
            bb = [float(x) for x in pred_obj[0][2:]]
            pred_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if(pred_obj[1] != 1):
                match_flag = 0
            if (pred_area >= Config.area_threshold) :
                pred_area_flag = 1 
            if pred_area >= Config.area_threshold  and pred_obj[1] != 1 :
                miss_flag = 1
		
        for true_obj in true_objects:	
            bbgt = [float(x) for x in true_obj[0][1:]]
            true_area = (bbgt[2] - bbgt[0]) * (bbgt[3] - bbgt[1])
            if(true_obj[1] != 1 ):
                match_flag = 0
            if (true_area >= Config.area_threshold):
                true_area_flag = 1		
            if true_area >= Config.area_threshold  and true_obj[1] != 1 :
                miss_flag = 1
		
        true_count = len(true_objects)
        pred_count = len(pred_objects)

        # number detected in ground truth array
        tos = sum([x[1] for x in true_objects])

        # number detected in pred array
        pos = sum([x[1] for x in pred_objects])
        # # 1 or 0 in label
        # is_prefect = (true_count == pred_count) and (
            # tos == pos) and (true_count == pos)
        # is_prefect = (pred_count == pos) and (true_count == tos)
        # if file_id=='COCO_val2014_000000000192': 
           # print("Fault:", pred_count, pos, true_count, tos)
           # exit()
        detected = [x[0] for x in pred_objects if x[1] == 1]
        extra_detections = [x[0] for x in pred_objects if x[1] == 0]
        missed = [x[0] for x in true_objects if x[1] == 0]
        # print(true_objects)
        # write all info to label file so that it can be used later if needed

        #####################
        # a) there is person and the network detects it
        # b) there is person and the network fails to detect it
        # c) there is no person and the network detection is right
        # d) there is no person and the network detection is wrong
        # e) there are many person but network detects only some -- Misses some

        # new_label = 0 if is_prefect == True else -1
        # if(new_label != 0):
            # if true_count > 0 and pred_count == 0:
                # new_label = 1
            # elif true_count == 0 and pred_count == 0:
                # new_label = 2
            # elif len(missed) > 0:
                # new_label = 3
            # elif len(extra_detections) > 0:
                # new_label = 4
        is_prefect = match_flag
        new_label = 0
        label = {"label": int(is_prefect),
                 "new_label": new_label,
                 "filename": file_id,
                 "gt_count": true_count,
                 "dr_count": pred_count,
                 "dr_match_count": tos,
                 "gt_match_count": pos,
                 "detected_list": detected,
                 "false_positive": extra_detections,
                 "missed": missed}

        all_labels.append(label)
        #creating labels for each file
        with open(os.path.join(label_dir, file_id+".json"), 'w') as f:
            json.dump(label, f)

        filename_for_list = file_id+'.jpg'
        
        if match_flag == 1 :
            if pred_count == 0:
                label_1_files_without_detections.append(filename_for_list)
            
            if pred_area_flag == 1 or true_area_flag == 1:
                label_1_files_with_detections.append(filename_for_list)
        
        if miss_flag == 1  :
            label_0_files.append(filename_for_list)

				
            # if(len(missed) > 0 and len(extra_detections) == 0 and len(detected) > 0):
                # label_0_missed_some.append(
                    # filename_for_list)  # Missed some of them

            # if(len(missed) > 0 and len(detected) == 0):  # no need to check for extra detection here
                # label_0_missed_all.append(filename_for_list)

            # if(len(extra_detections) > 0 and len(missed) == 0):
                # # detected extra - No person but network detected
                # label_0_false_positives.append(filename_for_list)

            # if(len(missed) > 0 and len(extra_detections) > 0):
                # # missed some and detected extra in some
                # label_0_missed_extra.append(filename_for_list)

            # # did not miss and no extra (THis should be empty)
            # if(len(missed) == 0 and len(extra_detections) == 0):
                # label_0_no_missed_no_extra.append(filename_for_list)
    #Saving Labels for all files
    save_file_list('l0', label_0_files)
    # save_file_list('l1', label_1_files_with_detections +
                   # label_1_files_without_detections)
    # save_file_list('l1_without_detections', label_1_files_without_detections)
    save_file_list('l1_with_detections', label_1_files_with_detections)

    # save_file_list('l0_missed_some', label_0_missed_some)
    # save_file_list('l0_missed_all', label_0_missed_all)

    # save_file_list('l0_fp', label_0_false_positives)
    # save_file_list('l0_missed_fp', label_0_missed_extra)
    # save_file_list('l0_no_miss_no_fp', label_0_no_missed_no_extra)

    with open(os.path.join(Config.cs_output_root, "all_labels.json"), 'w') as f:
        json.dump(all_labels, f)

def make_detector_labels_all(separate_root=None):
    """Creates a label file"""
    label_dir = Config.labels_dir
    gt_dir = Config.gt_dir
    dr_dir = Config.dr_dir
    print(label_dir, gt_dir, dr_dir)
    
    label_0_files = []

    label_1_files_with_detections = []
    label_1_files_without_detections = []

    label_0_missed_some = []
    label_0_missed_all = []

    label_0_false_positives = []
    label_0_missed_extra = []

    label_0_no_missed_no_extra = []  # This should be empty if everything is allright

    all_labels = []
    label_1_files_with_chair_detections = []
    label_1_files_with_person_detections = []
    label_1_files_with_personchair_detections = []
    label_0_files_with_chair = []
    label_0_files_with_person = []
    label_0_files_with_personchair = []

    if(separate_root != None):
        label_dir = os.path.join(separate_root, 'labels')
        gt_dir = os.path.join(separate_root, 'ground-truth')
        dr_dir = os.path.join(separate_root, 'detection-results')
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)

    ground_truth_files_list = glob.glob(gt_dir + '/*.txt')
    detection_files_list = glob.glob(dr_dir + '/*.txt')

    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    detection_files_list.sort()

    for txt_file in tqdm(ground_truth_files_list, 'Creating labels'):
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)  # gt file contents
        pred_file = os.path.join(dr_dir, file_id+".txt")
        pred_lines_list = file_lines_to_list(pred_file)  # dr file contents

        true_objects = [[l.split(), 0] for l in lines_list]
        pred_objects = [[l.split(), 0] for l in pred_lines_list]
        match_indices = []
        
        
        for pred_obj in pred_objects:
            bb = [float(x) for x in pred_obj[0][2:]]
            for true_obj in true_objects:
                if pred_obj[0][0] == true_obj[0][0]:  # same class
                    bbgt = [float(x) for x in true_obj[0][1:]]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(
                        bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                    
                    if(ov >= Config.threshold):
                        true_obj[1] = 1
                        pred_obj[1] = 1
                        

        match_flag = 1
       	chair_flag = 1
        person_flag = 1		
        for pred_obj in pred_objects:
            if(pred_obj[1] != 1):
                match_flag = 0
            if(pred_obj[0][0] != 'chair'):
                chair_flag = 0
            # if(pred_obj[0][0] != 'person'):
                # person_flag = 0
            
        for true_obj in true_objects:	
            if(true_obj[1] != 1 ):
                match_flag = 0
            if(true_obj[0][0] != 'chair'):
                chair_flag = 0
            # if(true_obj[0][0] != 'person'):
                # person_flag = 0
        true_count = len(true_objects)
        pred_count = len(pred_objects)

        # number detected in ground truth array
        tos = sum([x[1] for x in true_objects])

        # number detected in pred array
        pos = sum([x[1] for x in pred_objects])
        detected = [x[0] for x in pred_objects if x[1] == 1]
        extra_detections = [x[0] for x in pred_objects if x[1] == 0]
        missed = [x[0] for x in true_objects if x[1] == 0]
        
        # write all info to label file so that it can be used later if needed

        is_prefect = match_flag
        new_label = 0
        label = {"label": int(is_prefect),
                 "new_label": new_label,
                 "filename": file_id,
                 "gt_count": true_count,
                 "dr_count": pred_count,
                 "dr_match_count": tos,
                 "gt_match_count": pos,
                 "detected_list": detected,
                 "false_positive": extra_detections,
                 "missed": missed}

        all_labels.append(label)
        #creating labels for each file
        with open(os.path.join(label_dir, file_id+".json"), 'w') as f:
            json.dump(label, f)

        filename_for_list = file_id+'.jpg'
        
        if match_flag == 1 :
            
            if pred_count == 0:
                label_1_files_without_detections.append(filename_for_list)
            else:
                label_1_files_with_detections.append(filename_for_list)
                if (chair_flag == 1):
                    label_1_files_with_chair_detections.append(filename_for_list)
                if (person_flag == 1):
                    label_1_files_with_person_detections.append(filename_for_list)
                if (person_flag == 1 and chair_flag == 1):
                   label_1_files_with_personchair_detections.append(filename_for_list)
        else:
            label_0_files.append(filename_for_list)
            if (chair_flag == 1):
                   label_0_files_with_chair.append(filename_for_list)
            if (person_flag == 1):
                   label_0_files_with_person.append(filename_for_list)
            if (person_flag == 1 and chair_flag == 1):
                   label_0_files_with_personchair.append(filename_for_list)
    #Saving Labels for all files
    save_file_list('l0', label_0_files)
    save_file_list('l0_chair', label_0_files_with_chair)
    save_file_list('l0_person', label_0_files_with_person)
    save_file_list('l0_personchair', label_0_files_with_personchair)
    # save_file_list('l1', label_1_files_with_detections +
                   # label_1_files_without_detections)
    # save_file_list('l1_without_detections', label_1_files_without_detections)
    save_file_list('l1_with_detections', label_1_files_with_detections)
    save_file_list('l1_with_chair_detections', label_1_files_with_chair_detections)
    save_file_list('l1_with_person_detections', label_1_files_with_person_detections)
    save_file_list('l1_with_personchair_detections', label_1_files_with_personchair_detections)

    # save_file_list('l0_missed_some', label_0_missed_some)
    # save_file_list('l0_missed_all', label_0_missed_all)

    # save_file_list('l0_fp', label_0_false_positives)
    # save_file_list('l0_missed_fp', label_0_missed_extra)
    # save_file_list('l0_no_miss_no_fp', label_0_no_missed_no_extra)

    with open(os.path.join(Config.cs_output_root, "all_labels.json"), 'w') as f:
        json.dump(all_labels, f)

def make_detector_labels_OR(separate_root=r'/home/local2/Ferdous/YOLO/Datasets/val/chair/outputs'):
    """Creates a label file"""
    label_dir = Config.labels_dir
    gt_dir = Config.gt_dir
    dr_dir = Config.dr_dir

    label_0_files = []

    label_1_files_with_detections = []
    label_1_files_without_detections = []

    label_0_missed_some = []
    label_0_missed_all = []

    label_0_false_positives = []
    label_0_missed_extra = []

    label_0_no_missed_no_extra = []  # This should be empty if everything is allright

    all_labels = []

    if(separate_root != None):
        label_dir = os.path.join(separate_root, 'labels')
        gt_dir = os.path.join(separate_root, 'ground-truth')
        dr_dir = os.path.join(separate_root, 'detection-results')
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)

    ground_truth_files_list = glob.glob(gt_dir + '/*.txt')
    detection_files_list = glob.glob(dr_dir + '/*.txt')

    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    detection_files_list.sort()

    for txt_file in tqdm(ground_truth_files_list, 'Creating labels'):
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)  # gt file contents
        pred_file = os.path.join(dr_dir, file_id+".txt")
        pred_lines_list = file_lines_to_list(pred_file)  # dr file contents

        true_objects = [[l.split(), 0] for l in lines_list]
        pred_objects = [[l.split(), 0] for l in pred_lines_list]
        match_indices = []

        # # count1=count2=0
        for pred_obj in pred_objects:
            bb = [float(x) for x in pred_obj[0][2:]]
            # # count1+=1 
            # # count2=0
            for true_obj in true_objects:
                # # count2+=1
                # # # the prediction already matches with another gt
                # # if(true_obj[1] == 1):
                    # # continue
                if pred_obj[0][0] == true_obj[0][0]:  # same class
                    bbgt = [float(x) for x in true_obj[0][1:]]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(
                        bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                    # # print(count1,count2,ov)
                    if(ov >= Config.threshold):
                        true_obj[1] = 1
                        pred_obj[1] = 1
                        # # break

        true_count = len(true_objects)
        pred_count = len(pred_objects)

        # number detected in ground truth array
        tos = sum([x[1] for x in true_objects])

        # number detected in pred array
        pos = sum([x[1] for x in pred_objects])
        # 1 or 0 in label
        is_prefect = (true_count == pred_count) and (
            tos == pos) and (true_count == pos)
        is_prefect = (pred_count == pos) and (true_count == tos)
        # # if file_id=='COCO_val2014_000000000192': 
           # # print("Fault:", pred_count, pos, true_count, tos)
           # # exit()
        detected = [x[0] for x in pred_objects if x[1] == 1]
        extra_detections = [x[0] for x in pred_objects if x[1] == 0]
        missed = [x[0] for x in true_objects if x[1] == 0]
        # print(true_objects)
        # write all info to label file so that it can be used later if needed

        #####################
        # a) there is person and the network detects it
        # b) there is person and the network fails to detect it
        # c) there is no person and the network detection is right
        # d) there is no person and the network detection is wrong
        # e) there are many person but network detects only some -- Misses some

        new_label = 0 if is_prefect == True else -1
        if(new_label != 0):
            if true_count > 0 and pred_count == 0:
                new_label = 1
            elif true_count == 0 and pred_count == 0:
                new_label = 2
            elif len(missed) > 0:
                new_label = 3
            elif len(extra_detections) > 0:
                new_label = 4

        label = {"label": int(is_prefect),
                 "new_label": new_label,
                 "filename": file_id,
                 "gt_count": true_count,
                 "dr_count": pred_count,
                 "dr_match_count": tos,
                 "gt_match_count": pos,
                 "detected_list": detected,
                 "false_positive": extra_detections,
                 "missed": missed}

        all_labels.append(label)

        with open(os.path.join(label_dir, file_id+".json"), 'w') as f:
            json.dump(label, f)

        filename_for_list = file_id+'.jpg'
        if(is_prefect):
            if true_count == 0 and pred_count == 0:
                label_1_files_without_detections.append(filename_for_list)
            else:
                label_1_files_with_detections.append(filename_for_list)
        else:
            label_0_files.append(filename_for_list)

            if(len(missed) > 0 and len(extra_detections) == 0 and len(detected) > 0):
                label_0_missed_some.append(
                    filename_for_list)  # Missed some of them

            if(len(missed) > 0 and len(detected) == 0):  # no need to check for extra detection here
                label_0_missed_all.append(filename_for_list)

            if(len(extra_detections) > 0 and len(missed) == 0):
                # detected extra - No person but network detected
                label_0_false_positives.append(filename_for_list)

            if(len(missed) > 0 and len(extra_detections) > 0):
                # missed some and detected extra in some
                label_0_missed_extra.append(filename_for_list)

            # did not miss and no extra (THis should be empty)
            if(len(missed) == 0 and len(extra_detections) == 0):
                label_0_no_missed_no_extra.append(filename_for_list)

    save_file_list('l0', label_0_files)
    save_file_list('l1', label_1_files_with_detections +
                   label_1_files_without_detections)
    save_file_list('l1_without_detections', label_1_files_without_detections)
    save_file_list('l1_with_detections', label_1_files_with_detections)

    save_file_list('l0_missed_some', label_0_missed_some)
    save_file_list('l0_missed_all', label_0_missed_all)

    save_file_list('l0_fp', label_0_false_positives)
    save_file_list('l0_missed_fp', label_0_missed_extra)
    save_file_list('l0_no_miss_no_fp', label_0_no_missed_no_extra)

    with open(os.path.join(Config.cs_output_root, "all_labels.json"), 'w') as f:
        json.dump(all_labels, f)


def min_area_detected():
    Config.aug_dir_name = os.path.join('no_aug', 'person')
    detection_file_list = glob.glob(Config.dr_dir + '/*.txt')

    if len(detection_file_list) == 0:
        error("Error: No detection files found at {}".format(
            Config.dr_original_dir))

    detection_file_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}
    min_area = 9999999
    min_area_file = ''
    areas = []
    areas_small = []
    file_list_small = []

    original_list = file_lines_to_list(
        './data/nlstats/original_'+Config.dt+'.txt')
    original_list = [x.strip() for x in original_list]

    for txt_file in tqdm(detection_file_list, 'Scanning detection files'):
        # print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        if not file_id in original_list:
            continue

        lines_list = file_lines_to_list(txt_file)
        added = False
        for l in lines_list:
            pred_split = l.split(' ')
            bb = [float(x) for x in pred_split[2:]]
            area = (bb[2]*416 - bb[0]*416) * \
                (bb[3]*416 - bb[1]*416)/(416.0 * 416.0)

            areas.append(area)
            if area < min_area:
                min_area = area
                min_area_file = txt_file

            if(area < 0.060327293):
                areas_small.append(areas)
                if(added == False):
                    file_list_small.append(txt_file)
                added = True

    print('Min area is', min_area)
    print('The file is ', min_area_file)
    print(len(areas_small))
    print(len(file_list_small))

    with open('./data/file_list_dr_small', 'w') as fl:
        fl.write('\n'.join(file_list_small))

    return set([os.path.basename(f).split('.')[0]+'\n' for f in file_list_small])

    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    num_bins = 10
    n, bins, patches = plt.hist(
        areas_small, num_bins, facecolor='blue', alpha=0.5)
    plt.show()


def main(_):
    # filter_GT()
    filter_GT(Config.class_names)
    filter_DR(Config.class_names)
    make_detector_labels_all()
    #filter_GT(Config.class_names)
    #filter_DR(Config.class_names)
    #make_detector_labels(os.path.join(FLAGS.output_root_dir, 'specific', 'person'))

    # check_is_in_detection('COCO_train2014_000000013279',
    #   'chair 0.359890615940094 0.08738887906074524 0.5872656464576721 0.2040277481079102 0.0')


def make_labelfor_single_image(gt_file, dr_file):
    """Creates a label file"""

    lines_list = file_lines_to_list(gt_file)  # gt file contents
    pred_lines_list = file_lines_to_list(dr_file)  # dr file contents

    true_objects = [[l.split(), 0, -1, 0.0] for l in lines_list]
    pred_objects = [[l.split(), 0, -1, 0.0] for l in pred_lines_list]
    match_indices = []
    matches = 0

    for pi, pred_obj in enumerate(pred_objects):
        # if pred_obj[1] == 1:
        #     continue
        bb = [float(x) for x in pred_obj[0][2:]]

        for ti, true_obj in enumerate(true_objects):
            # the prediction already matches with another gt
            if(true_obj[1] == 1):
                continue
            ov = 0.0
            maxiou = 0

            if pred_obj[0][0] == true_obj[0][0]:  # same class
                bbgt = [float(x) for x in true_obj[0][1:]]

                # # determine the (x, y)-coordinates of the intersection rectangle
                # xA = max(bb[0], bbgt[0])
                # yA = max(bb[1], bbgt[1])
                # xB = min(bb[2], bbgt[2])
                # yB = min(bb[3], bbgt[3])
                # # compute the area of intersection rectangle
                # interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                # # compute the area of both the prediction and ground-truth
                # # rectangles
                # boxAArea = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                # boxBArea = (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                # # compute the intersection over union by taking the intersection
                # # area and dividing it by the sum of prediction + ground-truth
                # # areas - the interesection area
                # iou = interArea / float(boxAArea + boxBArea - interArea)
                # print(iou)

                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(
                    bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                      + 1) * (bbgt[3] - bbgt[1] + 1) - (iw * ih)
                    ov = (iw * ih) / ua

                print(ov)

                if(ov > true_obj[3]):
                    true_obj[3] = ov

                if(ov > pred_obj[3]):
                    pred_obj[3] = ov

                if(ov >= Config.threshold):
                    matches += 1
                    # true_obj[3] = ov

                    # true_obj[1] = 1
                    # pred_obj[1] = 1
                    # true_obj[2] = pi
                    # pred_obj[2] = ti
                    # break
            else:
                print("Label did not match")

        for ti, true_obj in enumerate(true_objects):
            if true_obj[1] == 1:
                continue

            if true_obj[3] == pred_obj[3] and true_obj[3] >= Config.threshold:
                true_obj[1] = 1
                pred_obj[1] = 1
                true_obj[2] = pi
                pred_obj[2] = ti
            else:
                true_obj[1] = 0
                true_obj[3] = 0.0

    print("Matches ", matches)
    pp = pprint.PrettyPrinter(depth=5, compact=True)

    pp.pprint(true_objects)
    print('-------------\n')
    pp.pprint(pred_objects)

    true_count = len(true_objects)
    pred_count = len(pred_objects)
    # number detected in ground truth array
    tos = sum([x[1] for x in true_objects])
    # number detected in pred array
    pos = sum([x[1] for x in pred_objects])
    # 1 or 0 in label
    is_prefect = (true_count == pred_count) and (
        tos == pos) and (true_count == pos)
    detected = [x[0] for x in pred_objects if x[1] == 1]
    extra_detections = [x[0] for x in pred_objects if x[1] == 0]
    missed = [x[0] for x in true_objects if x[1] == 0]

    # print(true_objects)
    # write all info to label file so that it can be used later if needed
    label = {"label": int(is_prefect),
             "filename": 'file_id',
             "gt_count": true_count,
             "dr_count": pred_count,
             "dr_match_count": tos,
             "gt_match_count": pos,
             "detected_list": detected,
             "false_positive": extra_detections,
             "missed": missed,
             "gt_per_obj_label": true_objects}

    return label


def check_if_has_crowd(class_names):
    ground_truth_files_list = glob.glob(Config.gt_original_dir + '/*.txt')

    print(Config.gt_original_dir)

    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()

    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    crowd_file_list = set()

    for txt_file in tqdm(ground_truth_files_list, 'Finding is crowd'):
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)

        for line in lines_list:
            class_name, left, top, right, bottom, iscrowd = line.split()
            width = float(right) - float(left)
            height = float(bottom) - float(top)
            area = width * height
            if(class_name in class_names and iscrowd == '1.0'):
                crowd_file_list.add(file_id+'\n')

    with open(os.path.join('./data/nlstats', "all_crowd_train.txt"), 'w') as ofile:
        ofile.writelines(crowd_file_list)

    return crowd_file_list


def get_original_training_list(dt='val'):

    load_path = '/home/bijay/Dropbox/CESGM_project/Bijay/DatasetWithNewCode/' + \
        dt+'/person/without_bbox_no_augment.dat'

    if dt == 'val':
        load_path = '/home/bijay/Dropbox/CESGM_project/Bijay/DatasetWithNewCode/' + \
            dt+'/person/without_bbox.dat'

    x, y, meta = pickle_load_save.load(load_path)

    print(meta[0])
    original_list = set()

    for m in tqdm(meta):
        original_list.add(m[3]+'\n')

    print(len(y))
    return original_list, x, y, meta


def get_difference(type='val'):
    crowd_list = check_if_has_crowd(Config.class_names)
    original_list, x, y, meta = get_original_training_list(Config.dt)
    crowded_in_original = crowd_list.intersection(original_list)

    crowded_0 = set()
    crowded_1 = set()

    for im in crowded_in_original:
        for index, m in enumerate(meta):
            if(im.strip() == m[3].strip()):
                if(y[index] == 0):
                    crowded_0.add(im)
                else:
                    crowded_1.add(im)

    img_base_bath = Config.root_images_dir

    with open(os.path.join('./data/nlstats', "crowd_"+Config.dt+".txt"), 'w') as ofile:
        ofile.writelines(crowded_in_original)
    with open(os.path.join('./data/nlstats', "crowd_"+Config.dt+"_0.txt"), 'w') as ofile:
        ofile.writelines(crowded_0)
    with open(os.path.join('./data/nlstats', "crowd_"+Config.dt+"_1.txt"), 'w') as ofile:
        ofile.writelines(crowded_1)


def draw_images():
    make_images_from_file(os.path.join('./data/nlstats', "crowd_" +
                                       Config.dt+"_0.txt"), 'crowd_0', k=20, gt_dir=Config.gt_original_dir)
    make_images_from_file(os.path.join('./data/nlstats', "crowd_" +
                                       Config.dt+"_1.txt"), 'crowd_1', k=20, gt_dir=Config.gt_original_dir)


def draw_images_small():
    make_images_from_file(os.path.join('./data/nlstats', "small_" +
                                       Config.dt+"_0.txt"), 'small_0', k=40, gt_dir=Config.gt_dir)
    make_images_from_file(os.path.join('./data/nlstats', "small_" +
                                       Config.dt+"_1.txt"), 'small_1', k=40, gt_dir=Config.gt_dir)


def make_images_from_file(filename, image_type, k=30, gt_dir=Config.gt_dir):
    lstFilenames = file_lines_to_list(filename)

    random_images = random.choices(lstFilenames, k=k)
    opdir = './data/nlstats/images/'+image_type

    if not os.path.exists(opdir):
        os.makedirs(opdir)
    for img in random_images:
        di.draw_img_test_file(img.strip(), opdir, gt_dir)


def get_common_small(type='val'):

    all_small_set = min_area_detected()
    print(len(all_small_set))

    original_list, x, y, meta = get_original_training_list(Config.dt)

    small_in_original = all_small_set.intersection(original_list)

    small_0 = set()
    small_1 = set()

    for im in small_in_original:
        for index, m in enumerate(meta):
            if(im.strip() == m[3].strip()):
                if(y[index] == 0):
                    small_0.add(im)
                else:
                    small_1.add(im)

    with open(os.path.join('./data/nlstats', "small_"+Config.dt+".txt"), 'w') as ofile:
        ofile.writelines(small_in_original)

    with open(os.path.join('./data/nlstats', "small_"+Config.dt+"_0.txt"), 'w') as ofile:
        ofile.writelines(small_0)

    with open(os.path.join('./data/nlstats', "small_"+Config.dt+"_1.txt"), 'w') as ofile:
        ofile.writelines(small_1)


def make_original_list(dt):
    original_list, x, y, meta = get_original_training_list(Config.dt)

    with open('./data/nlstats/original_'+dt+'.txt', 'w') as f:
        f.write('\n'.join([x[3] for x in meta]))


def list_without_crowd(dt):

    allCrowds = file_lines_to_list('./data/nlstats/crowd_'+dt+'.txt')
    original_list = file_lines_to_list('./data/nlstats/original_'+dt+'.txt')

    newList = set(original_list).difference(allCrowds)

    with open('./data/nlstats/dt_minus_crowd_'+dt+'.txt', 'w') as f:
        f.write('\n'.join([x.strip() for x in newList]))


def crowd_small_intersection():
    dt = Config.dt
    crowds_1 = file_lines_to_list('./data/nlstats/crowd_'+dt+'_1.txt')
    crowds_0 = file_lines_to_list('./data/nlstats/crowd_'+dt+'_0.txt')

    small_1 = file_lines_to_list('./data/nlstats/small_'+dt+'_1.txt')
    small_0 = file_lines_to_list('./data/nlstats/small_'+dt+'_0.txt')

    crowd_small_intersection = set(crowds_1).intersection(set(small_1))
    print('1: ', len(crowd_small_intersection))

    crowd_small_intersection = set(crowds_0).intersection(set(small_0))
    print('0: ', len(crowd_small_intersection))


if __name__ == "__main__":
    #Config.aug_dir_name = os.path.join('no_aug', 'person')
    # make_detector_labels_extra(Config.cs_output_root)
    app.run(main)
    # crowd_small_intersection()
    exit(0)

    # make_images_from_file(
    #     './data/nlstats/small_val_1.txt', 'small_1', gt_dir=Config.gt_dir)

    # di.draw_img_test_file('COCO_train2014_000000247190',
    #                       './data/nlstats', Config.gt_dir)

    # exit(0)

    # make_images_from_file(
    #     './data/label1.txt', 'label_1', gt_dir=Config.gt_dir)

    # make_images_from_file(
    #     './data/label0.txt', 'label_0', gt_dir=Config.gt_dir)

    # exit(0)

    # filter_DR(Config.class_names)
    filter_GT(Config.class_names)
    # make_detector_labels_extra(Config.cs_output_root)

    single_file_name = os.path.join(
        Config.gt_original_dir, 'COCO_train2014_000000247190.txt')
    output_dir = './data/nlstats/gt'

    filter_GT_single_file(Config.class_names, single_file_name, output_dir)

    # exit(0)

    # make_original_list(Config.dt)
    # list_without_crowd(Config.dt)

    # filter_GT_single_file(Config.class_names, '/media/bijay/Projects/Datasets/val/person/outputs/no_aug/person/detection-results/COCO_val2014_000000159402.txt',

    #                       )

    # get_common_small(Config.dt)
    # get_difference(Config.dt)

    # draw_images()

    # min_area_detected()
    draw_images_small()

    # exit(0)

    # filter_GT(Config.class_names)

    # # # filter_GT(Config.class_names)

    # min_area_detected()

    # file_id = 'COCO_val2014_000000000328'
    # file_id = 'COCO_val2014_000000542024'
    # file_id = 'COCO_val2014_000000565045'
    # gt_root = '/media/bijay/Projects/Datasets/val/person/outputs/no_aug/person/ground-truth'
    # dr_root = '/media/bijay/Projects/Datasets/val/person/outputs/no_aug/person/detection-results'

    # gt_file = os.path.join(gt_root, file_id+'.txt')
    # dr_file = os.path.join(dr_root, file_id+'.txt')
    # label = make_labelfor_single_image(gt_file, dr_file)
    # pp = pprint.PrettyPrinter(depth=5, compact=True)
    # pp.pprint(label)

    # di.draw_img_test_file(
    #     file_id, output_dir='./data/scratch', gt_dir=Config.gt_dir)

    # get_difference()
