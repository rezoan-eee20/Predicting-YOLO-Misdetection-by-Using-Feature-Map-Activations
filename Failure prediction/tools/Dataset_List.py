from PIL import Image
import glob
image_list = []
with open('ValDataset_List.txt', 'w') as f:
  for filename in glob.glob(r'C:\Users\SIU856512759\Desktop\Dataset\Val\images\*.jpg'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)
    print(filename)
    f.writelines(filename)
    f.writelines("\n")
    im.close()

f.close()