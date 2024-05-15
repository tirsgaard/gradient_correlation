import torch
from time import time

def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.device) -> None:
    """ Train the model for one epoch
    
    Args:
        model: the model to train
        optimizer: the optimizer to use
        criterion: the loss function to use
        train_loader: the data loader to use
        device: the device train on
    
    Returns:
        None
    """
    model.train()
    for i, (x, y) in enumerate(train_loader):
        t = time()
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        l = criterion(y_hat, y.float().squeeze(-1))
        l.backward()
        optimizer.step()
        
def validate(model: torch.nn.Module, criterion: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    """ Validate the model
    
    Args:
        model: the model to validate
        criterion: the loss function to use
        val_loader: the data loader to use
        device: the device to validate on
    
    Returns:
        val_loss: the validation loss
        val_accuracy: the validation accuracy
    """
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            l = criterion(y_hat, y.squeeze(-1).float())
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                y_hat = (y_hat > 0).float()
                val_accuracy += (y_hat == y).sum().item()
            else:
                val_accuracy += y[range(y.shape[0]), y_hat.softmax(-1).argmax(-1)].sum().item()
            
            val_loss += l.item()
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader.dataset)
    return val_loss, val_accuracy


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """ Test the model
    
    Args:
        model: the model to test
        test_loader: the data loader to use
        device: the device to test on
    
    Returns:
        test_accuracy: the test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            y_hat = (y_hat > 0).float()
            correct += (y_hat == y.float().view(-1, 1)).sum().item()
            total += y.size(0)
    test_accuracy = correct / total
    return test_accuracy

