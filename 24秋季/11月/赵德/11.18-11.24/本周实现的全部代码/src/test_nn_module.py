import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10("../datasets", False, transforms.ToTensor(), download=True)

my_loader = DataLoader(test_set,batch_size=4,shuffle=True,num_workers = 0,drop_last=False)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

input = 1
input = torch.tensor(input)
mynet = MyNet()
output = mynet(input)
print(output)