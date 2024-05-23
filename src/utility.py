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