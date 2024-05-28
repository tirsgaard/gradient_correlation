import torch

def entropy(P: torch.Tensor) -> torch.Tensor:
    """ Compute the entropy of n distributions
    Args:
        P: of shape (..., m) where m is the number of classes
    """
    return -torch.sum(P*torch.log(P), dim=-1)
    

def JS_div(P: torch.Tensor) -> torch.Tensor:
    """ Compute the Jensen-Shannon divergence between n distributions (equally weighted)
    Args:
        P: of shape (..., n, m) where n is the number of distributions and m is the number of classes
    """
    # Compute the average distribution
    P_avg = torch.mean(P, dim=-2)
    
    # Compute the KL divergence between each distribution and the average distribution
    JS = entropy(P_avg) - torch.mean(entropy(P), dim=-1)
    return JS


def cross_entropy_parallel(y_hat: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """ Compute the cross entropy between n distributions
    Args:
        y_hat: of shape (..., n, m) where n is the number of distributions and m is the number of classes. Target should be the same for all n.
        y_target: of shape (..., m) where m is the number of classes
    Returns:
        The cross entropy between the n distributions
    """
    n = y_hat.size(-2)
    # clone y_target n times
    y_target = y_target.unsqueeze(-2).expand_as(y_hat)
    # Vectorize parallel and batch dimensions
    y_hat = y_hat.view(-1, y_hat.size(-1))
    y_target = y_target.reshape(-1, y_target.size(-1))
    # Compute the cross entropy
    CE = torch.nn.functional.cross_entropy(y_hat, y_target, reduction='none')
    return CE.mean(-1)
    