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
        super().__init__(self)
        
        theta = torch.as_tensor(theta, dtype=torch.get_default_dtype())
        if trainable:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)

    def get_k(self, x: torch.Tensor, y: torch.Tensor):
        r2 = torch.sum(((x - y) / self.theta) ** 2)
        return torch.exp(-0.5 * r2)

    def get_k_fg(self, x: torch.Tensor, y: torch.Tensor, dim: int=None):
        """
        Returns d k(x,y) / d y
        shape: (D,)
        """
        k = self.get_k(x, y)
        r = x - y
        k_fg = k * r / self.theta**2
        if dim is not None:
            return k_fg[dim]
        else:
            return k_fg

    def get_k_gg(self, x: torch.Tensor, y: torch.Tensor, dims: List=None):
        k = self.get_k(x, y)
        r = x - y
        theta2 = self.theta**2

        D = x.shape[0]

        if theta2.ndim == 0:
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
