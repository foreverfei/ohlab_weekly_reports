import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from test_tensorboard import writer

test_set = torchvision.datasets.CIFAR10("../datasets", train = False, transform=torchvision.transforms.ToTensor(), download=True)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, img):
        return self.conv1(img)

my_loader = DataLoader(test_set,batch_size=64,shuffle=False,drop_last=False)

my_net = MyNet()
writer = SummaryWriter("../logs")
step = 0

for data in my_loader:
    imgs, labels = data
    output = my_net(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("imgs", imgs, step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step + 1