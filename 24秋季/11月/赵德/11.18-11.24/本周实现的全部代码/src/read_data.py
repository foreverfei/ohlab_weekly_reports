from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class MyData(Dataset):

    def __init__(self,root_dir,label_name):
        self.root_dir = root_dir
        self.label_name = label_name
        self.all_image_dir = os.path.join(root_dir,label_name)
        self.all_image_name = os.listdir(self.all_image_dir)

    def __getitem__(self,index):
        image_name = self.all_image_name[index]
        image_path = os.path.join(self.all_image_dir,image_name)
        totensor_resize_tool = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((100,100)),
            ]
        )
        image = totensor_resize_tool(Image.open(image_path))
        label = self.label_name
        return [image,label]

    def __len__(self):
        return len(self.all_image_name)

root_dir = "../dataset/hymenoptera_data/train"
ants_label_name = "ants"
bees_label_name = "bees"
ants_dataset = MyData(root_dir,ants_label_name)
bees_dataset = MyData(root_dir,bees_label_name)
train_dataset = ants_dataset + bees_dataset
image,label = ants_dataset[0]