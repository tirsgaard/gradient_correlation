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

def get_binary_MNIST(digits: tuple[int, int], val_split: float) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """ Function for loading the binary MNIST dataset with the specified digits and validation split
    Args:
        digits: The digits to classify
        val_split: The proportion of the training set to use for validation
        
        Returns:
            train_data: The training set
            val_data: The validation set
            test_data: The test set
    """
    # Load the data
    train_data = get_MNIST_train()
    test_data = get_MNIST_test()

    # Only keep the 0s and 1s for binary classification
    digits = (1,7)
    train_data = list(filter(lambda i: i[1] in digits, train_data))
    test_data = list(filter(lambda i: i[1] in digits, test_data))
    # Set 0 to be the negative class
    train_data = [(x, 0 if y == digits[0] else 1) for x, y in train_data]
    test_data = [(x, 0 if y == digits[0] else 1) for x, y in test_data]

    # Split the training set into training and validation
    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
    
    return train_data, val_data, test_data