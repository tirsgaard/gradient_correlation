import torch
from src.models.NN_models import SimpleMLP, SingleLayerMLP, Parallel_MLP
from datasets.download_MNIST import get_MNIST
from src.correlation_gradient import get_gradient, construct_covariance_matrix, svd_pseudo_inverse
from time import time
import matplotlib.pyplot as plt

def MSELoss_batch(y_hat, y):
    return 0.5*(y_hat-y).pow(2).sum(-1)

def maximum_determinant_ranking(full_matrix: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """ Iteratively tries to find sequence that maximises the determinant of the subeset of the matrix.
    
    Args:
        full_matrix: the full matrix to use [n_dim, n_dim]
        indexes: the indexes that have been added to the subset [n_subset]
        
    Returns:
        ranked_samples: the ranked samples
    """
    max_det = 0
    A = full_matrix[indexes][:, indexes]
    not_added_indexes = torch.ones(full_matrix.shape[0], dtype=bool)
    not_added_indexes[indexes] = False
    not_added_indexes = torch.arange(full_matrix.shape[0])[not_added_indexes]
    B_all = full_matrix[indexes][:, not_added_indexes].T
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
    return best_index

def condition_on_observations_solve(covariance_matrix: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """ Condition the covariance / correlation matrix of a list of observations.
    If the correlation between two indexes is above the cor_cutoff, the first index is removed.
    If is is_corr, the matrix is assumed to be a correlation matrix and the output will be a correlation matrix.
    
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        indexes: the list of indexes to condition on
        
    Returns:
        new_correlation_matrix: the new correlation matrix
    """
    
    indexes_not_used = [i for i in range(covariance_matrix.shape[0]) if i not in indexes]
    sigma11 = covariance_matrix[indexes_not_used][:, indexes_not_used]
    sigma22 = covariance_matrix[indexes][:, indexes]
    sigma12 = covariance_matrix[indexes_not_used][:, indexes]
    sigma21 = covariance_matrix[indexes][:, indexes_not_used]
    sigma_cond = sigma12 @ torch.linalg.solve(sigma22, sigma21)
    new_correlation_matrix = sigma11 - sigma_cond
    return new_correlation_matrix

def condition_on_observations_svd(covariance_matrix: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """ Condition the covariance / correlation matrix of a list of observations.
    If the correlation between two indexes is above the cor_cutoff, the first index is removed.
    If is is_corr, the matrix is assumed to be a correlation matrix and the output will be a correlation matrix.
    
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        indexes: the list of indexes to condition on
        
    Returns:
        new_correlation_matrix: the new correlation matrix
    """
    
    indexes_not_used = [i for i in range(covariance_matrix.shape[0]) if i not in indexes]
    sigma11 = covariance_matrix[indexes_not_used][:, indexes_not_used]
    sigma22 = covariance_matrix[indexes][:, indexes]
    sigma12 = covariance_matrix[indexes_not_used][:, indexes]
    sigma21 = covariance_matrix[indexes][:, indexes_not_used]
    sigma22_inv = svd_pseudo_inverse(sigma22, int((sigma22.shape[0]*0.9+1)//1))
    sigma_cond = sigma12 @ sigma22_inv @ sigma21
    new_correlation_matrix = sigma11 - sigma_cond
    return new_correlation_matrix

def condition_on_observations_inv(covariance_matrix: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """ Condition the covariance / correlation matrix of a list of observations.
    If the correlation between two indexes is above the cor_cutoff, the first index is removed.
    If is is_corr, the matrix is assumed to be a correlation matrix and the output will be a correlation matrix.
    
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        indexes: the list of indexes to condition on
        
    Returns:
        new_correlation_matrix: the new correlation matrix
    """
    
    indexes_not_used = [i for i in range(covariance_matrix.shape[0]) if i not in indexes]
    sigma11 = covariance_matrix[indexes_not_used][:, indexes_not_used]
    sigma22 = covariance_matrix[indexes][:, indexes]
    sigma12 = covariance_matrix[indexes_not_used][:, indexes]
    sigma21 = covariance_matrix[indexes][:, indexes_not_used]
    sigma22_inv = torch.linalg.inv(sigma22)
    sigma_cond = sigma12 @ sigma22_inv @ sigma21
    new_correlation_matrix = sigma11 - sigma_cond
    return new_correlation_matrix

def condition_on_observations_chol(covariance_matrix: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """ Condition the covariance / correlation matrix of a list of observations.
    If the correlation between two indexes is above the cor_cutoff, the first index is removed.
    If is is_corr, the matrix is assumed to be a correlation matrix and the output will be a correlation matrix.
    
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        indexes: the list of indexes to condition on
        
    Returns:
        new_correlation_matrix: the new correlation matrix
    """
    
    indexes_not_used = [i for i in range(covariance_matrix.shape[0]) if i not in indexes]
    sigma11 = covariance_matrix[indexes_not_used][:, indexes_not_used]
    sigma22 = covariance_matrix[indexes][:, indexes]
    sigma12 = covariance_matrix[indexes_not_used][:, indexes]
    sigma21 = covariance_matrix[indexes][:, indexes_not_used]
    L, info = torch.linalg.cholesky_ex(sigma22)
    if info != 0:
        raise RuntimeError("Cholesky factorization failed")
        
    sigma_cond = sigma12 @ torch.cholesky_solve(sigma21, L)
    new_correlation_matrix = sigma11 - sigma_cond
    return new_correlation_matrix

def iterative_condition_on_observation(iterated_covariance_matrix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """ 
    
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        index: the list of indexes to condition on. If multiple elements the last one is used
        
    Returns:
        new_correlation_matrix: the new correlation matrix
    """
    indexes_not_used = [i for i in range(iterated_covariance_matrix.shape[0]) if i not in index]
    sigma11 = iterated_covariance_matrix[indexes_not_used][:, indexes_not_used]
    sigma22 = iterated_covariance_matrix[index, index]
    sigma12 = iterated_covariance_matrix[indexes_not_used][:, index][:, None]
    sigma21 = iterated_covariance_matrix[index][indexes_not_used][None, :]
    sigma_cond = sigma12 @ (sigma21 / sigma22)
    new_correlation_matrix = sigma11 - sigma_cond
    return new_correlation_matrix


def iterative_condition_on_observation_faster(iterated_covariance_matrix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """ 
    
    Args:
        correlation_matrix: the correlation matrix to use [n_dim, n_dim]
        index: the list of indexes to condition on. If multiple elements the last one is used
        
    Returns:
        new_correlation_matrix: the new correlation matrix
    """
    indexes_not_used = [i for i in range(iterated_covariance_matrix.shape[0]) if i not in index]
    sigma11 = iterated_covariance_matrix[indexes_not_used][:, indexes_not_used]
    sigma22 = iterated_covariance_matrix[index, index]
    sigma12 = iterated_covariance_matrix[indexes_not_used][:, index]
    sigma21 = iterated_covariance_matrix[index][indexes_not_used]
    sigma_cond = torch.outer(sigma12, (sigma21 / sigma22))
    new_correlation_matrix = sigma11 - sigma_cond
    return new_correlation_matrix

def rank_covariance_information(covariance_matrix: torch.Tensor,  condition_method: callable, cutoff_number: int = 200) -> torch.Tensor:
    """ Rank the samples by information
    Args:
        covariance_matrix: the correlation matrix to use [n_dim, n_dim]
        
    Returns:
        ranked_samples: the ranked samples
    """
    n_dim = covariance_matrix.shape[0]
    ranked_samples = torch.zeros(n_dim, dtype=int)
    existing_indexes = torch.ones(n_dim, dtype=bool)
    current_covariance_matrix = covariance_matrix
    for i in range(n_dim):
        local_index = current_covariance_matrix.diag().argmax()
        index = torch.arange(n_dim)[existing_indexes][local_index]  # Get the index in the original correlation matrix
        ranked_samples[i] = index
        existing_indexes[index] = False
        
        if (condition_method == iterative_condition_on_observation) or condition_method == (iterative_condition_on_observation_faster):
            current_covariance_matrix = condition_method(current_covariance_matrix, local_index)
        else:
            current_covariance_matrix = condition_method(covariance_matrix, ranked_samples[:(i + 1)])
        if current_covariance_matrix.isnan().any() or i>=cutoff_number:
            # Add remaining indexes to ranked samples
            ranked_samples[i+1:] = torch.arange(n_dim)[existing_indexes]
            break
    return ranked_samples


def run_experiment(n_datapoints: int, n_parameters: int, n_layers: int, batch_size: int, device: torch.device, conditioning_methods: list[callable], cutoff_number: int = 200) -> torch.Tensor:
    """ This could be a random matrix, but to get a more representive covariance matrix we construct a model and use MNIST data."""
    model = SimpleMLP(input_size=784, output_size=10, hidden_size=n_parameters, num_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.)
    # Load the data
    train_data, val_data, test_data = get_MNIST(0.5, device=device)
    train_data = torch.utils.data.Subset(train_data, range(n_datapoints))
    # Shuffle the train data
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False)
    x = torch.cat([xs for xs, _ in train_loader], 0)

    grads = get_gradient(model, x, MSELoss_batch, optimizer, True, flatten=True, kernel="pKernel")
    covariance_matrix = construct_covariance_matrix(grads)
    print("Condition number of covariance matrix:", torch.linalg.cond(covariance_matrix))
    
    rankings = []
    timings = []
    with torch.no_grad():
        for method in conditioning_methods:
            t = time()
            ranked_samples = rank_covariance_information(covariance_matrix, method, cutoff_number)
            timings.append(time()-t)
            rankings.append(ranked_samples)
    timings = torch.tensor(timings)
    rankings = torch.stack(rankings)
    return timings, rankings
    


if __name__ == '__main__':
    # Setup
    n_datapoints = 1000
    n_parameters = 32
    n_layers = 4
    cutoff_number = 800
    do_exp = True
    
    # Other parameteres
    batch_size = 32
    
    names = ["Solve", "Inv", "SVD", "Chol", "iterative_condition", "iterative_condition_faster"]
    conditioning_methods = [condition_on_observations_solve, condition_on_observations_inv, condition_on_observations_svd, condition_on_observations_chol, iterative_condition_on_observation, iterative_condition_on_observation_faster]

    # Sample data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if do_exp:
        timings, rankings = run_experiment(n_datapoints, n_parameters, n_layers, batch_size, device, conditioning_methods, cutoff_number)
        # save timings and rankings
        torch.save(timings, "timings.pt")
        torch.save(rankings, "rankings.pt")
    else:
        timings = torch.load("timings.pt")
        rankings = torch.load("rankings.pt")
        
    for i, name in enumerate(names):
        print(f"Method: {name}, Time: {timings[i]}")
    
    
    # Calculate agreement of rankings
    n_rankings = min(rankings.shape[1], cutoff_number)
    agreement = torch.zeros(len(names), n_rankings)
    for i in range(len(names)):
        name_mask = torch.ones(len(names), dtype=bool)
        name_mask[i] = False
        for j in range(n_rankings):
            agreement[i, j] = (rankings[i][j] == rankings[name_mask, :(j+1)]).float().sum() >= 1.
    agreements = agreement.cumsum(1)/torch.arange(1, n_rankings+1)
    
    # Plot the agreement
    fig, ax = plt.subplots()
    for i, name in enumerate(names):
        ax.plot(agreements[i], label=name)    
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Mean agreement")
    plt.legend()
    plt.show()
    
    
    
    