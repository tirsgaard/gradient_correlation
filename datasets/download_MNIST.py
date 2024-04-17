import torch
import torchvision
from torchvision import transforms



def get_MNIST_train() -> torchvision.datasets.MNIST:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    return trainset

def get_MNIST_test() -> torchvision.datasets.MNIST:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
    return testset