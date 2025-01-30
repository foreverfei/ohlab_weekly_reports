import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch import reshape
from torch.utils.tensorboard import SummaryWriter

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=3,ceil_mode=True)

    def forward(self, img):
        return self.maxpool2d(img)

class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=3,dilation=2,ceil_mode=True)

    def forward(self, img):
        return self.maxpool2d(img)

img = torch.tensor([[122, 237, 0, 131, 12],
              [0, 15, 2, 33, 111],
              [1, 2, 1, 0, 0],
              [5, 2, 3, 1, 1],
              [2, 1, 0, 1, 1]])

img = torch.reshape(img, (1, 1, 5, 5))

print(img.shape)

my_net = MyNet()

img = my_net(img)
mynet2 = MyNet2()

test_set = torchvision.datasets.CIFAR10("../datasets", train=False, transform = torchvision.transforms.ToTensor(), download=True)

my_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

writer = SummaryWriter('../logs')
writer.add_image("maxpool2d",img, 0,dataformats='NCHW')

step = 0
for data in my_loader:
    imgs, labels = data
    output1 = my_net(imgs)
    output2 = mynet2(imgs)
    writer.add_images("maxpool2d_imgs", imgs, step)
    writer.add_images("maxpool2d_output", output1, step)
    writer.add_images("maxpool2d_output2", output2, step)
    step += 1