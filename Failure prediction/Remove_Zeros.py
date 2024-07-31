import os

folder_path = r'/home/local2/Ferdous/YOLO/Datasets/val/person/outputs/ground-truth'

for filename in os.listdir(folder_path):
    old_file_path = os.path.join(folder_path, filename)
    new_file_path = os.path.join(folder_path, str(int(filename.split(".")[0]))) + "." + filename.split(".")[1]
    os.rename(old_file_path, new_file_path)
