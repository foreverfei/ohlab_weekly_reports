import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, stride = 1, padding = 0, dilation = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        return x

mynet = MyNet()

test_set = torchvision.datasets.CIFAR10("../datasets", train = False, transform = transforms.ToTensor(), download = True)

my_loader = DataLoader(test_set, batch_size = 64, shuffle = True, num_workers = 0, drop_last = False)

writer = SummaryWriter("../logs")
step = 0

for data in my_loader:
    imgs, labels = data
    output = mynet(imgs)
    writer.add_images("ReLU", imgs, step)