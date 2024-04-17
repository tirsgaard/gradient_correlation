import torch


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """ Train the model for one epoch
    
    Args:
        model: the model to train
        optimizer: the optimizer to use
        criterion: the loss function to use
        train_loader: the data loader to use
    
    Returns:
        None
    """
    model.train()
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(x)
        l = criterion(y_hat, y.float().view(-1, 1))
        l.backward()
        optimizer.step()
        
def validate(model: torch.nn.Module, criterion: torch.nn.Module, val_loader: torch.utils.data.DataLoader) -> tuple[float, float]:
    """ Validate the model
    
    Args:
        model: the model to validate
        criterion: the loss function to use
        val_loader: the data loader to use
    
    Returns:
        val_loss: the validation loss
        val_accuracy: the validation accuracy
    """
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for x, y in val_loader:
            y_hat = model(x)
            l = criterion(y_hat, y.float().view(-1, 1))
            y_hat = (y_hat > 0).float()
            val_accuracy += (y_hat == y.float().view(-1, 1)).sum().item()
            val_loss += l.item()
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader.dataset)
    return val_loss, val_accuracy


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
    """ Test the model
    
    Args:
        model: the model to test
        test_loader: the data loader to use
    
    Returns:
        test_accuracy: the test accuracy
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = model(x)
            y_hat = (y_hat > 0).float()
            correct += (y_hat == y.float().view(-1, 1)).sum().item()
            total += y.size(0)
    test_accuracy = correct / total
    return test_accuracy