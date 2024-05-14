from typing import Iterable, Optional, Union
from sklearn.decomposition import PCA
from scipy.spatial import distance
import torch

def average_loss(y_pred: torch.Tensor, loss: torch.nn.Module) -> torch.Tensor:
    """ Compute the average loss using the probabilities
    Args:
        y_pred: the predictions [batch_size, n_classes]
        
    Returns:
        loss: the average loss
    """
    total_loss = 0.
    for i in range(y_pred.shape[-1]):
        total_loss += loss(y_pred.float(), torch.tensor([i], dtype=torch.long).repeat(y_pred.shape[0])) * y_pred.softmax(-1)[:, i]
    return total_loss

def get_gradient(model: torch.nn.Module, x: torch.Tensor, loss_fn: callable, opt: torch.optim.Optimizer, positive: bool = True, flatten: bool = False, use_label: bool = False, y: Optional[torch.Tensor] = None, pKernel: bool=False) -> torch.Tensor:
    """
    Get the gradient of each element of the batch in x of the model with respect to the loss function
    
    Args:
        model: the model to use
        x: the input data [N_samples, ...]
        y: the target data [N_samples, ...]
        loss_fn: the loss function to use
        opt: the optimizer to use
        positive: whether to perturb the loss function positively or negatively
        flatten: whether to flatten the gradient
        
    Returns:
        grads: the gradients of the model with respect to the loss function
    """
    batch_gradients = []
    device = next(model.parameters()).device
    x = x.to(device)
    if y is not None:
        y = y.to(device).float()
    
    def pkernel_loss(y_pred: torch.Tensor, loss: torch.nn.Module) -> torch.Tensor:
        total_loss = 0.
        n_classes = y_pred.shape[-1]
        for i in range(n_classes):
            y_target = torch.zeros_like(y_pred).scatter(1, torch.tensor([i], dtype=torch.long, device=y_pred.device).repeat(y_pred.shape[0]).unsqueeze(1), 1)
            total_loss += loss(y_pred.float(), y_target)
        total_loss = total_loss / n_classes**0.5
        return total_loss

    for i in range(len(x)):
        opt.zero_grad()
        y_pred = model(x[i, None, ...])
        if use_label:
            y_target = y[i, None]
        else :
            y_target = y_pred.detach().argmax(1)
            y_target = torch.zeros_like(y_pred).scatter(1, y_target.unsqueeze(1), 1)
       
        # get one-hot encoding of y_target
        if pKernel:
            loss = pkernel_loss(y_pred, loss_fn)
        else:
            loss = loss_fn(y_pred, y_target)
        loss.backward()
        grads = []
        for param in model.parameters():
                grads.append(param.grad.detach())
        if flatten:
            grads = torch.cat([grad.view(-1) for grad in grads])
        batch_gradients.append(grads)
    grads = torch.stack(batch_gradients)
    return grads

def get_gradient_batch(model: torch.nn.Module, data: Iterable[torch.Tensor], loss_fn: callable, opt: torch.optim.Optimizer, positive: bool = True) -> torch.Tensor:
    """
    Get the gradient of the batch in x of the model with respect to the loss function
    
    Args:
        model: the model to use
        data: the input data [N_saples][...]
        loss_fn: the loss function to use
        opt: the optimizer to use
        positive: whether to perturb the loss function positively or negatively
        
    Returns:
        grads: the gradients of the model with respect to the loss function
    """
    # Get the dimension of the parameters of the model
    n_params = sum(param.numel() for param in model.parameters())
    grads = torch.zeros(n_params)
        
    N_samples = 0
    opt.zero_grad()
    for x, y in data:
        x = x.to(next(model.parameters()).device)
        N_samples += x.shape[0]
        y_pred = model(x)
        loss = loss_fn(y_pred, y.float())
        loss.backward()
        grads_batch = []
        for param in model.parameters():
            grads_batch.append(param.grad.detach())
        grads_batch = torch.cat([grad.view(-1) for grad in grads_batch])
        grads_batch /= torch.norm(grads_batch)
        grads += grads_batch
    grads /= N_samples
    return grads

def construct_correlation_matrix(grads: torch.Tensor) -> torch.Tensor:
    """ Construct the correlation matrix from the gradients
    Args:
        grads: the gradients to use [batch_size, n_params]
        
    Returns:
        correlation_matrix: the correlation matrix [n_params, n_params]
    """
    normalised_gradients = grads - grads.mean(axis=0)[None, ...]
    normalised_gradients = normalised_gradients / (normalised_gradients.std(0)+10**-6)
    normalised_gradients[normalised_gradients.isnan()] = 0
    normalised_gradients = normalised_gradients/(torch.norm(normalised_gradients, dim=-1)[..., None]/(normalised_gradients.shape[-1]**0.5))
    correlation_matrix = torch.matmul(normalised_gradients, normalised_gradients.T)/normalised_gradients.shape[-1]
    return correlation_matrix


def maximum_determinant_ranking(full_matrix: torch.Tensor) -> torch.Tensor:
    """ Iteratively tries to find sequence that maximises the determinant of the subeset of the matrix.
    
    Args:
        full_matrix: the full matrix to use [n_dim, n_dim]
        
    Returns:
        ranked_samples: the ranked samples
    """
    indexes = torch.zeros(full_matrix.shape[0], dtype=int)
    added_indexes = torch.zeros(full_matrix.shape[0], dtype=bool)
    # Select point most informative (highest correlation)
    first_index = torch.argmax((full_matrix**2).sum(axis=0))
    indexes[0] = first_index
    added_indexes[first_index] = True
    
    stop_iteration = 300
    from tqdm import tqdm
    for i in tqdm(range(1, stop_iteration)):
        max_det = 0
        A = full_matrix[indexes[:i]][:, indexes[:i]]
        not_added_indexes = torch.arange(full_matrix.shape[0])[~added_indexes]
        B_all = full_matrix[indexes[:i]][:, not_added_indexes].T
        D = full_matrix[not_added_indexes, not_added_indexes]
        mat = A - B_all[..., None] @ B_all[..., None, :]/D[..., None, None]
        mat = (mat + torch.transpose(mat, 1, 2))/2  # Ensure symmetry
        all_mat = D*torch.linalg.det(mat)
        max_det, best_index = torch.max(all_mat, 0)
        
        # Vectorize the computation below
        """
        for j in range(full_matrix.shape[0]):
            if added_indexes[j]:
                continue
            new_indexes = torch.cat([indexes[:i], torch.tensor([j])])
            new_matrix = full_matrix[new_indexes][:, new_indexes]
            A = new_matrix[:-1, :-1]
            B = new_matrix[:-1, -1]
            D = new_matrix[-1, -1]
            det = D*torch.linalg.det(A - B[:, None]*B[None, :]/D)
            if det > max_det:
                max_det = det
                best_index = j
        """
        # Convert best_index to the original index
        best_index = not_added_indexes[best_index]
        indexes[i] = best_index
        assert not added_indexes[best_index]
        added_indexes[best_index] = True
    # Add remaining indexes
    indexes[stop_iteration:] = torch.arange(full_matrix.shape[0])[~added_indexes]
    return indexes
    
    

def construct_covariance_matrix(grads: torch.Tensor) -> torch.Tensor:
    """ Construct the covariance matrix from the gradients
    Args:
        grads: the gradients to use [batch_size, n_params]
        
    Returns:
        covariance_matrix: the covariance matrix [n_params, n_params]
    """
    normalised_gradients = grads - grads.mean(axis=0)[None, ...]
    length = torch.norm(normalised_gradients, dim=-1)
    normalised_gradients = normalised_gradients/length[..., None]
    correlation_matrix = torch.matmul(normalised_gradients, normalised_gradients.T)
    covariance_matrix = correlation_matrix*torch.outer(length, length)
    return covariance_matrix

def get_most_informative_index(correlation_matrix: torch.Tensor) -> int:
    """ Get the most informative index from the correlation matrix
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        
    Returns:
        index: the most informative index
    """
    information = (correlation_matrix**2).sum(axis=0)
    return torch.argmax(information)

def get_most_unique_index(correlation_matrix: torch.Tensor) -> int:
    """ Get the most unique index from the correlation matrix
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        
    Returns:
        index: the most unique index
    """
    information = (correlation_matrix**2).sum(axis=0)
    information[[correlation_matrix[i, i] == 0 for i in range(correlation_matrix.shape[0])]] = torch.inf
    return torch.argmin(information)

def condition_on_observation(correlation_matrix: torch.Tensor, index: int) -> torch.Tensor:
    """ Condition the correlation matrix on an observation
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        index: the index to condition on
        
    Returns:
        new_correlation_matrix: the new correlation matrix
    """
    n_dim = correlation_matrix.shape[0]
    corr_squared = correlation_matrix**2
    denom = torch.sqrt(((1-corr_squared[:, index])[:, None]*(1-corr_squared[index, :])[None, :]).clip(10**-6, float('inf')))
    new_correlation_matrix = (correlation_matrix - correlation_matrix[:, index][:, None]*correlation_matrix[index, :][None, :])/denom
    # Remove index from correlation matrix
    new_correlation_matrix = new_correlation_matrix[torch.arange(n_dim) != index][:, torch.arange(n_dim) != index]
    
    if torch.linalg.cond(new_correlation_matrix) > 10**4:
        new_correlation_matrix = new_correlation_matrix + torch.eye(new_correlation_matrix.shape[0])
        new_correlation_matrix = covariance2correlation(new_correlation_matrix)
    new_correlation_matrix = new_correlation_matrix.clip(-1, 1)
    return new_correlation_matrix

def condition_on_observations(correlation_matrix: torch.Tensor, indexes: torch.Tensor, cor_cutoff: float = 0.99) -> torch.Tensor:
    """ Condition the correlation matrix of a list of observations
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        indexes: the list of indexes to condition on
        
    Returns:
        new_correlation_matrix: the new correlation matrix
    """
    # Remove highly intercorrelated indexes before conditioning for numerical stability
    dropped_points = torch.zeros(len(indexes), dtype=bool)
    for i in range(len(indexes)):
        for j in range(i+1, len(indexes)):
            if correlation_matrix[indexes[i], indexes[j]].abs() > cor_cutoff:
                dropped_points[indexes[j]] = True
    
    indexes_not_used = [i for i in range(correlation_matrix.shape[0]) if i not in indexes]
    indexes = indexes[~dropped_points]  # Remove highly correlated indexes
    sigma11 = correlation_matrix[indexes_not_used][:, indexes_not_used]
    sigma22 = correlation_matrix[indexes][:, indexes]
    sigma12 = correlation_matrix[indexes_not_used][:, indexes]
    sigma21 = correlation_matrix[indexes][:, indexes_not_used]
    sigma_cond = sigma12 @ torch.linalg.solve(sigma22, sigma21)
    sigma_cond = (sigma_cond + sigma_cond.T)/2  # Ensure symmetry
    new_correlation_matrix = sigma11 - sigma_cond
    if torch.linalg.cond(new_correlation_matrix) > 10**4:
        new_correlation_matrix = new_correlation_matrix + torch.eye(new_correlation_matrix.shape[0])
    # Convert from covariance to correlation
    new_correlation_matrix = covariance2correlation(new_correlation_matrix)
    return new_correlation_matrix

def covariance2correlation(covariance_matrix: torch.Tensor) -> torch.Tensor:
    """ Convert a covariance matrix to a correlation matrix
    Args:
        covariance_matrix: the covariance matrix to convert [n_dim, n_dim]
        
    Returns:
        correlation_matrix: the correlation matrix [n_dim, n_dim]
    """
    diag = torch.sqrt(torch.diag(covariance_matrix)).clip(10**-6, float('inf'))
    correlation_matrix = covariance_matrix/(diag[:, None]*diag[None, :])
    return correlation_matrix

def rank_correlation_information(correlation_matrix: torch.Tensor) -> torch.Tensor:
    """ Rank the samples by information
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        
    Returns:
        ranked_samples: the ranked samples
    """
    n_dim = correlation_matrix.shape[0]
    ranked_samples = torch.zeros(n_dim, dtype=int)
    existing_indexes = torch.ones(n_dim, dtype=bool)
    for i in range(n_dim):
        local_index = get_most_informative_index(correlation_matrix)
        index = torch.arange(n_dim)[existing_indexes][local_index]  # Get the index in the original correlation matrix
        ranked_samples[i] = index
        existing_indexes[index] = False
        correlation_matrix = condition_on_observation(correlation_matrix, local_index)
        if correlation_matrix.isnan().any():
            # Add remaining indexes to ranked samples
            ranked_samples[i+1:] = torch.arange(n_dim)[existing_indexes]
            break
    return ranked_samples

def rank_correlation_uniqueness(x_train: torch.Tensor, x_unlabel: torch.Tensor, model: torch.nn.Module, loss_fn: callable, opt: torch.optim.Optimizer, positive: bool = True) -> torch.Tensor:
    """ Rank the samples by uniqueness (decorrelation) of the gradients from trianing data
    Args:
        x_train: the training data [N_train][...]
        x_unlabel: the unlabelled data [N_unlabel][...]
        model: the model to use
        loss_fn: the loss function to use
        opt: the optimizer to use
        positive: whether to perturb the loss function positively or negatively
        
    Returns:
        ranked_samples: the ranked samples
    """
    # Compile x into a single tensor
    x_train = torch.cat([xs for xs, _ in x_train], 0)
    train_gradient = get_gradient(model, x_train, loss_fn, opt, positive, flatten=True)
    train_gradient /= torch.norm(train_gradient, dim=-1)[..., None]
    train_gradient = train_gradient.sum(0)
    x_unlabel = torch.cat([xs for xs, _ in x_unlabel], 0)
    unlabel_gradients = get_gradient(model, x_unlabel, loss_fn, opt, positive, flatten=True)
    combined_gradients = torch.cat([train_gradient[None, ...], unlabel_gradients + train_gradient[None, ...]])
    correlation_matrix = construct_correlation_matrix(combined_gradients)
    return correlation_matrix[0].sort().indices


def rank_sample_information(x: Iterable[torch.Tensor], model: torch.nn.Module, loss_fn: callable, opt: torch.optim.Optimizer, positive: bool = True, pre_condition_index: Optional[torch.Tensor]=None) -> torch.Tensor:
    """ Rank the samples by information
    Args:
        x: the input data [N_samples][...]
        model: the model to use
        loss_fn: the loss function to use
        opt: the optimizer to use
        positive: whether to perturb the loss function positively or negatively
        pre_condition_index: the index to condition on in the beginning. Mostly used for trianing data. If None, no conditioning is done
        
    Returns:
        ranked_samples: the ranked samples
    """
    # Compile x into a single tensor
    x = torch.cat([xs for xs, _ in x], 0)
    
    grads = get_gradient(model, x, loss_fn, opt, positive, flatten= True, pKernel=True)
    correlation_matrix = construct_correlation_matrix(grads)    
    if len(pre_condition_index) > 0:
        correlation_matrix = condition_on_observations(correlation_matrix, pre_condition_index, cor_cutoff=0.8)
    # Construct new correlation matrix without indexes conditioned on
    non_condition_index = torch.tensor([i for i in range(len(x)) if i not in pre_condition_index])
    ranked_samples = rank_correlation_information(correlation_matrix)
    non_condition_index = non_condition_index[ranked_samples]
    # Combine ranked samples with pre-conditioned indexes
    ranked_samples = torch.cat([pre_condition_index, ranked_samples])
    return ranked_samples

def rank_pca_information(x: Iterable[torch.Tensor], model: torch.nn.Module, loss_fn: callable, opt: torch.optim.Optimizer, positive: bool = True, pre_condition_index: Optional[torch.Tensor]=None) -> torch.Tensor:
    """ Rank the samples by information
    Args:
        x: the input data [N_samples][...]
        model: the model to use
        loss_fn: the loss function to use
        opt: the optimizer to use
        positive: whether to perturb the loss function positively or negatively
        pre_condition_index: the index to condition on in the beginning. Mostly used for trianing data. If None, no conditioning is done
        
    Returns:
        ranked_samples: the ranked samples
    """
    # Compile x into a single tensor
    x = torch.cat([xs for xs, _ in x], 0)
    grads = get_gradient(model, x, loss_fn, opt, positive, flatten= True)
    correlation_matrix = construct_correlation_matrix(grads)    

    # Run PCA on correlation matrix and project x onto the first two principal components
    pca = PCA(n_components=1)
    pca.fit(correlation_matrix)
    x_pca = pca.transform(correlation_matrix)
    
    # Rank points according to the maximum distance of the minimum distance to other points
    distances = distance.cdist(x_pca, x_pca)
    distances[range(len(x_pca)), range(len(x_pca))] = float('inf')  # Set diagonal to infinity
    min_distances = distances[range(len(x_pca)), distances.argmin(axis=1)]
    ranked_samples = min_distances.argsort()[::-1].copy()
    return ranked_samples