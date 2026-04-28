import torch
import torch.nn as nn

from typing import List

__all__ = ["KernelFunction", "RBFKernelFunction"]

class KernelFunction(nn.Module):
    def __init__(self, theta: torch.Tensor, trainable=False):
        super().__init__()

    def get_k(self, x: torch.Tensor, y: torch.Tensor):
        pass

    def get_k_fg(self, x: torch.Tensor, y: torch.Tensor, dim: int=None):
        """
        Returns d k(x,y) / d y
        shape: (D,)
        """
        pass

    def get_k_gf(self, x: torch.Tensor, y: torch.Tensor, dim: int=None):
        """
        Returns d k(x,y) / d x
        shape: (D,)
        """
        if dim is not None:
             return -self.get_k_fg(x, y)[dim]
        else:
            return -self.get_k_fg(x, y)

    def get_k_gg(self, x: torch.Tensor, y: torch.Tensor, dims: List=None):
        pass

    def get_full_k(self, x: torch.Tensor, y: torch.Tensor):
        """
        Returns block kernel:

        [ k        dk/dy ]
        [ dk/dx    d2k/dxdy ]

        shape: (D+1, D+1)
        """
        k_ff = self.get_k(x, y).reshape(1, 1)
        k_fg = self.get_k_fg(x, y).reshape(1, -1)
        k_gf = self.get_k_gf(x, y).reshape(-1, 1)
        k_gg = self.get_k_gg(x, y)

        return torch.cat([
            torch.cat([k_ff, k_fg], dim=1),
            torch.cat([k_gf, k_gg], dim=1),
        ], dim=0)


class RBFKernelFunction(KernelFunction):
    def __init__(self, theta: torch.Tensor, trainable=False):
        super().__init__(theta, trainable=trainable)
        
        theta = torch.as_tensor(theta, dtype=torch.get_default_dtype())
        if trainable:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)

    def _theta_like(self, X: torch.Tensor):
        return self.theta.to(dtype=X.dtype, device=X.device)

    def _pairwise(self, X: torch.Tensor, Y: torch.Tensor):
        theta = self._theta_like(X)
        diff = X[:, None, :] - Y[None, :, :]
        r2 = torch.sum((diff / theta) ** 2, dim=-1)
        K = torch.exp(-0.5 * r2)
        return K, diff, theta

    def get_k_matrix(self, X: torch.Tensor, Y: torch.Tensor):
        K, _, _ = self._pairwise(X, Y)
        return K

    def get_k_fg_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        grad_dims: torch.Tensor = None,
    ):
        """
        Batched d k(x, y) / d y terms.

        If grad_dims is provided, returns shape (N, M), selecting one
        derivative dimension for each row in Y. Otherwise returns (N, M, D).
        """
        K, diff, theta = self._pairwise(X, Y)
        values = K[..., None] * diff / theta**2

        if grad_dims is None:
            return values

        grad_dims = grad_dims.to(device=X.device, dtype=torch.long)
        index = grad_dims.view(1, -1, 1).expand(X.shape[0], -1, 1)
        return values.gather(dim=2, index=index).squeeze(-1)

    def get_k_gf_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        grad_dims: torch.Tensor = None,
    ):
        """
        Batched d k(x, y) / d x terms.

        If grad_dims is provided, returns shape (N, M), selecting one
        derivative dimension for each row in X. Otherwise returns
        point-major shape (N * D, M).
        """
        K, diff, theta = self._pairwise(X, Y)
        values = -K[..., None] * diff / theta**2

        if grad_dims is not None:
            grad_dims = grad_dims.to(device=X.device, dtype=torch.long)
            index = grad_dims.view(-1, 1, 1).expand(-1, Y.shape[0], 1)
            return values.gather(dim=2, index=index).squeeze(-1)

        return values.permute(0, 2, 1).reshape(X.shape[0] * X.shape[1], Y.shape[0])

    def get_k_gg_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        x_grad_dims: torch.Tensor = None,
        y_grad_dims: torch.Tensor = None,
    ):
        """
        Batched mixed second derivative covariance terms.

        Shape conventions:
        - no grad dims: returns (N * D, M * D), point-major by dimension.
        - only y_grad_dims: returns (N * D, M).
        - only x_grad_dims: returns (N, M * D).
        - both grad dims: returns (N, M).
        """
        K, diff, theta = self._pairwise(X, Y)
        theta2 = theta**2
        inv_theta2 = 1.0 / theta2
        D = X.shape[1]

        if theta2.numel() == 1:
            eye_term = torch.eye(D, dtype=X.dtype, device=X.device) * inv_theta2
            outer_term = diff[..., :, None] * diff[..., None, :] * inv_theta2**2
        else:
            eye_term = torch.diag(inv_theta2)
            outer_term = (
                diff[..., :, None]
                * diff[..., None, :]
                * inv_theta2.view(1, 1, D, 1)
                * inv_theta2.view(1, 1, 1, D)
            )

        values = K[..., None, None] * (eye_term - outer_term)

        if x_grad_dims is None and y_grad_dims is None:
            return values.permute(0, 2, 1, 3).reshape(
                X.shape[0] * D, Y.shape[0] * D
            )

        if y_grad_dims is not None:
            y_grad_dims = y_grad_dims.to(device=X.device, dtype=torch.long)
            y_index = y_grad_dims.view(1, -1, 1, 1).expand(
                X.shape[0], -1, D, 1
            )
            values = values.gather(dim=3, index=y_index).squeeze(3)

            if x_grad_dims is None:
                return values.permute(0, 2, 1).reshape(X.shape[0] * D, Y.shape[0])

            x_grad_dims = x_grad_dims.to(device=X.device, dtype=torch.long)
            x_index = x_grad_dims.view(-1, 1, 1).expand(-1, Y.shape[0], 1)
            return values.gather(dim=2, index=x_index).squeeze(2)

        x_grad_dims = x_grad_dims.to(device=X.device, dtype=torch.long)
        x_index = x_grad_dims.view(-1, 1, 1, 1).expand(-1, Y.shape[0], 1, D)
        values = values.gather(dim=2, index=x_index).squeeze(2)
        return values.reshape(X.shape[0], Y.shape[0] * D)

    def get_k(self, x: torch.Tensor, y: torch.Tensor):
        theta = self._theta_like(x)
        r2 = torch.sum(((x - y) / theta) ** 2)
        return torch.exp(-0.5 * r2)

    def get_k_fg(self, x: torch.Tensor, y: torch.Tensor, dim: int=None):
        """
        Returns d k(x,y) / d y
        shape: (D,)
        """
        k = self.get_k(x, y)
        r = x - y
        theta = self._theta_like(x)
        k_fg = k * r / theta**2
        if dim is not None:
            return k_fg[dim]
        else:
            return k_fg

    def get_k_gg(self, x: torch.Tensor, y: torch.Tensor, dims: List=None):
        k = self.get_k(x, y)
        r = x - y
        theta2 = self._theta_like(x)**2

        D = x.shape[0]

        if theta2.numel() == 1:
            # isotropic
            eye_term = torch.eye(D, dtype=x.dtype, device=x.device) / theta2
            outer_term = torch.outer(r, r) / (theta2**2)
        else:
            # anisotropic
            eye_term = torch.diag(1.0 / theta2)
            outer_term = torch.outer(r / theta2, r / theta2)

        k_gg = k * (eye_term - outer_term)
        if dims is not None and len(dims)==2:
            return k_gg[dims[0], dims[1]]
        else:
            return k_gg
