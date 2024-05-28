import torch
import torch.nn.functional as F
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

def validate_parallel(model: torch.nn.Module, criterion: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
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
                correct = y.argmax(-1)[..., None] == y_hat.softmax(-1).argmax(-1)
                val_accuracy += correct.float().mean(-1).sum().item()
            
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

def train_IVON(model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.device) -> None:
    """ Train the model using the IVON method
    Args:
        model: the model to train
        optimizer: the optimizer to use
        criterion: the loss function to use
        train_loader: the data loader to use
        device: the device to train on
    
    Returns:
    """
    model.train()
    for i, (x, y) in enumerate(train_loader):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            l = criterion(y_hat, y.float().squeeze(-1))
            l.backward()
        optimizer.step()

def validate_IVON(model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device, test_samples: int = 1) -> tuple[float, float]:
    """ Validate the model using the IVON method
    
    Args:
        model: the model to validate
        optimizer: the optimizer to use
        criterion: the loss function to use
        val_loader: the data loader to use
        device: the device to validate on
        test_samples: the number of samples to use for testing
    
    Returns:
        val_loss: the validation loss
        val_accuracy: the validation accuracy
    """
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for x, y in val_loader:
            sampled_probs = []
            losses = []
            for i in range(test_samples):
                with optimizer.sampled_params():
                    sampled_logit = model(x)
                    losses.append(criterion(sampled_logit, y.float().squeeze(-1)))
                    sampled_probs.append(F.softmax(sampled_logit, dim=1))
            prob = torch.mean(torch.stack(sampled_probs), dim=0)
            val_loss += torch.mean(torch.stack(losses), dim=0)
            _, prediction = prob.max(1)
            correct = (prediction == y.argmax(1))
            val_accuracy += correct.sum().item()
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader.dataset)
    return val_loss, val_accuracy
