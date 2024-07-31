import os
from os.path import join
import logging
import itertools
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ConfigDTS:
    root_dir = '/home/local2/Ferdous/YOLO/Datasets'
    class_name_file = '/home/local2/Ferdous/YOLO/yolov3_tf2/data/coco.names'
    weights = './checkpoints/yolov3.tf'

    # class_name_file = '/home/siu855576252/projects/gm/data/coco.names'
    # root_dir = '/scratch/siu855576252/datasets/coco'
    # weights = '/home/siu855576252/projects/gm/checkpoints/yolov3.tf'

    tiny = False
    size = 416
    num_classes = 80

    threshold = 0.5
    area_threshold = 0.060327293  # Normalized area

    # selected_classes = ['person']

    def __init__(self, dataset_type, class_names):
        self.dt = dataset_type  # training or validation
        self.annotation_dir_name = 'annotations'
        self.annotation_year = '2014'
        self.output_dir_name = 'output_s'
        self.file_list_dir_name = 'file_list'
        self.tfrecord_dir_name = 'tfrecords'
        self._gt_dir_name = 'ground-truth'
        self._dr_dir_name = 'detection-results'
        self._image_dir_name = 'images-optional'
        self._output_data_dir_name = 'data'
        self._gt_original_dir_name = 'ground-truth-original'
        self._dr_original_dir_name = 'detection-results-original'
        self.cs_output_dir_name = 'outputs'
        self._labels_dir_name = 'labels'
        self.aug_dir_name = ''
        self.images_dir_name = 'images'
        self.class_names = class_names # change this to class_names later
        self.selected_classes = class_names
        self.cs_dir_name = self.get_cs_dir_name(self.class_names)
        self.classes = self.class_name_file
        self.detector_data_name = 'data.dat'
        self.detector_unbalanced_data_name = 'unbalanced_data.dat'

        self.initialize_dirs()

    @property
    def dataset_dir(self):
        return join(self.root_dir, self.dt)

    @property
    def root_annotation_file(self):
        return join(self.root_dir, self.annotation_dir_name, 'instances'+'_'+self.dt+self.annotation_year+'.json')

    @property
    def root_images_dir(self):
        return join(self.root_dir, self.dt + self.annotation_year)

    @property
    def root_output_dir(self):
        return join(self.root_dir, self.output_dir_name)

    # CLASS SPECIFIC FUNCTIONS -------------------------------------
    @property
    def output_dir(self):
        return join(self.root_dir, self.dt)

    @property
    def file_list_dir(self):
        return join(self.dataset_dir, self.cs_dir_name, self.file_list_dir_name)

    @property
    def annotation_dir(self):
        return join(self.dataset_dir, self.cs_dir_name, self.annotation_dir_name)

    @property
    def tfrecord_dir(self):
        return join(self.dataset_dir, self.cs_dir_name, self.tfrecord_dir_name)

    @property
    def cs_output_root(self):
        return join(self.dataset_dir, self.cs_dir_name, self.cs_output_dir_name, self.aug_dir_name)

    @property
    def gt_dir(self):
        return join(self.cs_output_root, self._gt_dir_name)

    # Unfiltered GT_DIRECTORY
    @property
    def gt_original_dir(self):
        return join(self.cs_output_root, self._gt_original_dir_name)

    # Unfiltered DR_DIRECTORY
    @property
    def dr_original_dir(self):
        return join(self.cs_output_root, self._dr_original_dir_name)

    @property
    def dr_dir(self):
        return join(self.cs_output_root, self._dr_dir_name)

    @property
    def img_optional_dir(self):
        return join(self.cs_output_root, self._image_dir_name)

    @property
    def output_data_dir(self):
        return join(self.cs_output_root, self._output_data_dir_name)

    @property
    def labels_dir(self):
        return join(self.cs_output_root, self._labels_dir_name)

    @property
    def cs_file_list_dir(self):
        return join(self.cs_output_root, self.file_list_dir_name)

    @property
    def cs_images_dir(self):
        return join(self.cs_output_root, self.images_dir_name)

    @property
    def detector_data_file(self):
        return join(self.cs_output_root, self.detector_data_name)
    # Helper funtions

    def initialize_dirs(self):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

    def get_cs_dir_name(self, class_names):
        # Class name should have atleast one element
        assert(len(class_names) > 0)
        return '_'.join(class_names)

    def get_all_class_names_dict(self):
        classes_combination = {}
        classes_combination[0] = []
        classes_combination[1] = self.class_names
        for i in range(2, len(self.class_names)+1):
            classes_combination[i] = list(itertools.combinations(self.class_names, i))
        return classes_combination

    def get_all_class_names_combinations(self):
        classes_combination = self.get_all_class_names_dict()
        all_class_name_combination = classes_combination[1]

        for i in range(2, len(classes_combination.keys())):
            comb = ['and'.join(x) for x in classes_combination[i]]
            all_class_name_combination = all_class_name_combination + comb

        return all_class_name_combination

    def reset_all_vars(self):
        self.annotation_dir_name = 'annotations'
        self.annotation_year = '2014'
        self.output_dir_name = 'output_s'
        self.file_list_dir_name = 'file_list'
        self.tfrecord_dir_name = 'tfrecords'
        self._gt_dir_name = 'ground-truth'
        self._dr_dir_name = 'detection-results'
        self._image_dir_name = 'images-optional'
        self._output_data_dir_name = 'data'
        self._gt_original_dir_name = 'ground-truth-original'
        self._dr_original_dir_name = 'detection-results-original'
        self.cs_output_dir_name = 'outputs'
        self._labels_dir_name = 'labels'
        self.aug_dir_name = ''
        self.images_dir_name = 'images'
        self.cs_dir_name = self.get_cs_dir_name(self.class_names)
        self.classes = self.class_name_file

        self.initialize_dirs()


Config = ConfigDTS('val', ['person']) # 'person', 'car', 'chair', 'cup'
