from pycocotools.coco import COCO
import json

# path to the COCO annotations file
annotations_file = "/home/local2/Ferdous/YOLO/annotation/annotations/instances_train2014.json"

# path to the output file with only person class annotations
output_file = "/home/local2/Ferdous/YOLO/Datasets/train/annotations/cup_annotation.json"

# initialize COCO api
coco = COCO(annotations_file)

# get category ids for person class
catIds = coco.getCatIds(catNms=['cup'])

# get annotations for person class
annIds = coco.getAnnIds(catIds=catIds)
annotations = coco.loadAnns(annIds)

# get images with person class annotations
imgIds = [ann['image_id'] for ann in annotations]
imgIds = list(set(imgIds))
images = coco.loadImgs(imgIds)

# create new COCO instance with only person class annotations
coco_person = COCO()
coco_person.dataset = {
    "info": coco.dataset["info"],
    "licenses": coco.dataset["licenses"],
    "images": images,
    "annotations": annotations,
    "categories": coco.loadCats(catIds)
}
coco_person.createIndex()

# save the output file
with open(output_file, 'w') as f:
    json.dump(coco_person.dataset, f)
