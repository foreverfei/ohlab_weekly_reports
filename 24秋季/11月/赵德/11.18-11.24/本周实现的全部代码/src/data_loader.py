from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from read_data import train_dataset

writer = SummaryWriter("../logs")

my_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last = True)
step = 0
for data in my_loader:
    imgs,labels = data
    # print(imgs.shape)
    # print(labels)
    writer.add_images("test_loader",imgs,step)
    step = step + 1

writer.close()