from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter



writer = SummaryWriter("../logs")
image_path = "data/train/ants_image/0013035.jpg"
pil_image = Image.open(image_path)
np_image = cv2.imread(image_path)
to_tensor = transforms.ToTensor()
tensor_image = to_tensor(pil_image)
writer.add_image("tensor_image", tensor_image, 0)
writer.close()