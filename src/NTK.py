import torch
from typing import Iterable
from src.correlation_gradient import get_gradient

def svd_pseudo_inverse(matrix: torch.Tensor, k: int) -> torch.Tensor:
    u, s, v = torch.svd(matrix)
    s_inv = 1/s
    s_inv[k:] = 0
    return v @ torch.diag(s_inv) @ u.T

class GaussianFit(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device: torch.device, kernel_method: str = "direct", noise_var: float = 0.0):
        super(GaussianFit, self).__init__()
        self.device = device
        self.model = model
        self.kernel_method = kernel_method
        self.noise_var = noise_var
        self.covariance_matrix = None
        self.optimizer = None
        
    def fit(self, data: Iterable[torch.Tensor], optimizer: torch.optim.Optimizer, loss_batched: torch.nn.Module):
        self.optimizer = optimizer
        self.loss = loss_batched
        xs, ys, y_hats = [], [], []
        with torch.no_grad():
            for x, y in data:
                xs.append(x)
                ys.append(y)
                y_hats.append(self.model(x.to(self.device)))
        xs = torch.cat(xs, 0).to(self.device)
        y = torch.cat(ys, 0).to(self.device)
        y_hat = torch.cat(y_hats, 0).to(self.device)
        label_diff = y - y_hat
        self.grads = get_gradient(self.model, xs, loss_batched, self.optimizer, True, True, y=y, kernel = self.kernel_method)
        #self.grads = self.grads - self.grads.mean(-1, keepdim=True)
        self.covarinace_kernel = self.grads@self.grads.T
        self.covarinace_matrix = self.covarinace_kernel.clone()
        self.covarinace_matrix[range(self.covarinace_matrix.shape[0]), range(self.covarinace_matrix.shape[0])] += self.noise_var
        self.W = torch.linalg.solve(self.covarinace_matrix.cpu(), label_diff.cpu()).to(self.device)
        #self.W = svd_pseudo_inverse(self.covarinace_matrix, 100) @ label_diff
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_grad = get_gradient(self.model, x, self.loss, self.optimizer, True, True, kernel = self.kernel_method)
        #x_grad = x_grad - x_grad.mean(-1, keepdim=True)
        K_xX = x_grad@self.grads.T
        with torch.no_grad():
            y_hat = self.model(x) #torch.softmax(self.model(x), -1)
        return y_hat + K_xX @ self.W