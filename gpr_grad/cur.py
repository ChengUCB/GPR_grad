import torch

__all__ = ["CUR_deterministic", "CUR_deterministic_step"]

def CUR_deterministic(
    X: torch.Tensor,
    n_col: int,
    error_estimate: bool = True,
    costs=1.0,
):
    """
    Deterministic CUR-style column/row selection for a covariance matrix.

    Parameters
    ----------
    X : torch.Tensor, shape (N, N)
        Input covariance/similarity matrix.
    n_col : int
        Number of columns/rows to select.
    error_estimate : bool
        Whether to compute residual error after each selection.
    costs : float or torch.Tensor, shape (N,)
        Optional cost weights for each column.

    Returns
    -------
    rsel : torch.LongTensor, shape (n_col,)
        Selected indices.
    rerror : torch.Tensor, shape (n_col,)
        Residual errors.
    """
    RX = X.clone()

    device = X.device
    dtype = X.dtype

    rsel = torch.empty(n_col, dtype=torch.long, device=device)
    rerror = torch.zeros(n_col, dtype=dtype, device=device)

    if not torch.is_tensor(costs):
        costs = torch.tensor(costs, dtype=dtype, device=device)
    else:
        costs = costs.to(dtype=dtype, device=device)

    for i in range(n_col):
        sel, RX = CUR_deterministic_step(RX, n_col - i, costs)
        rsel[i] = sel

        if error_estimate:
            rerror[i] = torch.sum(torch.abs(RX))

    return rsel, rerror


def CUR_deterministic_step(
    cov: torch.Tensor,
    k: int,
    costs=1.0,
):
    """
    One deterministic CUR selection step.

    Uses top-k eigenvectors of cov, selects the row/column with largest
    leverage weight, then orthogonalizes the covariance against that row.

    Parameters
    ----------
    cov : torch.Tensor, shape (N, N)
        Current residual covariance matrix.
    k : int
        Number of eigenvectors to use.
    costs : float or torch.Tensor, shape (N,)

    Returns
    -------
    sel : torch.LongTensor scalar
        Selected index.
    rcov : torch.Tensor, shape (N, N)
        Updated residual covariance matrix.
    """
    device = cov.device
    dtype = cov.dtype
    N = cov.shape[0]

    if not torch.is_tensor(costs):
        costs = torch.tensor(costs, dtype=dtype, device=device)
    else:
        costs = costs.to(dtype=dtype, device=device)

    # For symmetric covariance matrices.
    # eigh returns eigenvalues in ascending order.
    evals, evecs = torch.linalg.eigh(cov)

    k = min(k, N)
    top_evecs = evecs[:, -k:]

    weights = torch.sum(top_evecs**2, dim=1) / costs
    sel = torch.argmax(weights)

    vsel = cov[sel].clone()
    vnorm = torch.linalg.norm(vsel)

    if vnorm < torch.finfo(dtype).eps:
        rcov = cov.clone()
        return sel, rcov

    vsel = vsel / vnorm

    # Orthogonalize each row against vsel:
    #
    # rcov[i] = cov[i] - vsel * dot(cov[i], vsel)
    #
    proj = cov @ vsel
    rcov = cov - proj[:, None] * vsel[None, :]

    return sel, rcov
