import torchvision

train_set = torchvision.datasets.CIFAR10("../datasets", True, download=True)
test_set = torchvision.datasets.CIFAR10("../datasets", False, download=True)