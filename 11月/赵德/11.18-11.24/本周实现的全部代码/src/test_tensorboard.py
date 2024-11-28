from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os

# from read_data import ants_dataset, bees_dataset, train_dataset


class Mydata(Dataset):

    def __init__(self,root_path,label_name):
        self.root_path = root_path
        self.label_name = label_name
        self.all_image_path = os.path.join(self.root_path,self.label_name)
        self.all_image_name = os.listdir(self.all_image_path)

    def __getitem__(self,index):
        image_name = self.all_image_name[index]
        image_path = os.path.join(self.all_image_path,image_name)
        image = Image.open(image_path)
        label = self.label_name
        return image,label

    def __len__(self):
        return len(self.all_image_name)

ants_root_path = "data/train"
ants_label_name = "ants_image"
ants_dataset = Mydata(ants_root_path,ants_label_name)
bees_root_path = "data/train"
bees_label_name = "bees_image"
bees_dataset = Mydata(bees_root_path,bees_label_name)
train_dataset = ants_dataset + bees_dataset

writer = SummaryWriter("../logs")

for i in range(10):
    PIL_image,label = train_dataset[i]
    np_image = np.array(PIL_image)
    writer.add_image("test", np_image,i,dataformats="HWC")
for i in range(100):
    writer.add_scalar("y=x*x*x",i*i*i,i)
writer.close()