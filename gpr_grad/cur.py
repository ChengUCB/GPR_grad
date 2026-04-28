import torch

__all__ = ["CUR_deterministic", "CUR_deterministic_step"]

def CUR_deterministic(
    X: torch.Tensor,
    n_col: int,
    error_estimate: bool = True,
    costs=1.0,
    eigensolver: str = "auto",
    lobpcg_niter: int = 40,
    lobpcg_tol: float = None,
    exact_size: int = 512,
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
    eigensolver : {"auto", "exact", "lobpcg"}
        How to compute top eigenvectors for leverage weights. "exact" uses
        torch.linalg.eigh and reproduces the original behavior. "lobpcg" uses
        an iterative top-k solver. "auto" uses exact solves for small or
        unsuitable problems and LOBPCG when k is small relative to N.
    lobpcg_niter : int
        Maximum LOBPCG iterations for the iterative eigensolver.
    lobpcg_tol : float or None
        Optional LOBPCG convergence tolerance.
    exact_size : int
        In "auto" mode, use exact eigendecomposition when N <= exact_size.

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
        sel, RX = CUR_deterministic_step(
            RX,
            n_col - i,
            costs,
            eigensolver=eigensolver,
            lobpcg_niter=lobpcg_niter,
            lobpcg_tol=lobpcg_tol,
            exact_size=exact_size,
        )
        rsel[i] = sel

        if error_estimate:
            rerror[i] = torch.sum(torch.abs(RX))

    return rsel, rerror


def CUR_deterministic_step(
    cov: torch.Tensor,
    k: int,
    costs=1.0,
    eigensolver: str = "auto",
    lobpcg_niter: int = 40,
    lobpcg_tol: float = None,
    exact_size: int = 512,
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
    eigensolver : {"auto", "exact", "lobpcg"}
        Eigenvector method for leverage-score weights.
    lobpcg_niter : int
        Maximum LOBPCG iterations for the iterative eigensolver.
    lobpcg_tol : float or None
        Optional LOBPCG convergence tolerance.
    exact_size : int
        In "auto" mode, use exact eigendecomposition when N <= exact_size.

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

    k = min(k, N)
    top_evecs = _top_eigenvectors(
        cov,
        k,
        eigensolver=eigensolver,
        lobpcg_niter=lobpcg_niter,
        lobpcg_tol=lobpcg_tol,
        exact_size=exact_size,
    )

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


def _top_eigenvectors(
    cov: torch.Tensor,
    k: int,
    eigensolver: str,
    lobpcg_niter: int,
    lobpcg_tol: float,
    exact_size: int,
):
    eigensolver = eigensolver.lower()
    if eigensolver in ("eigh", "full"):
        eigensolver = "exact"

    if eigensolver not in ("auto", "exact", "lobpcg"):
        raise ValueError(
            "eigensolver must be one of {'auto', 'exact', 'lobpcg'}"
        )

    N = cov.shape[0]
    if k <= 0:
        return cov.new_empty(N, 0)

    if eigensolver == "exact" or _should_use_exact(N, k, eigensolver, exact_size):
        return _top_eigenvectors_exact(cov, k)

    try:
        return _top_eigenvectors_lobpcg(cov, k, lobpcg_niter, lobpcg_tol)
    except RuntimeError:
        if eigensolver == "lobpcg":
            raise
        return _top_eigenvectors_exact(cov, k)


def _should_use_exact(N: int, k: int, eigensolver: str, exact_size: int):
    if eigensolver != "auto":
        return False

    # torch.lobpcg requires N >= 3 * k, and exact is faster for small matrices.
    return N <= exact_size or N < 3 * k


def _top_eigenvectors_exact(cov: torch.Tensor, k: int):
    # For symmetric covariance matrices. eigh returns eigenvalues ascending.
    _, evecs = torch.linalg.eigh(cov)
    return evecs[:, -k:]


def _top_eigenvectors_lobpcg(
    cov: torch.Tensor,
    k: int,
    lobpcg_niter: int,
    lobpcg_tol: float,
):
    # Match torch.linalg.eigh's lower-triangle convention on residuals that may
    # have drifted away from exact symmetry after row projection.
    A = torch.tril(cov)
    A = A + torch.tril(cov, diagonal=-1).T

    X = _lobpcg_initial_guess(A.shape[0], k, A.dtype, A.device)
    _, evecs = torch.lobpcg(
        A,
        k=k,
        X=X,
        niter=lobpcg_niter,
        tol=lobpcg_tol,
        largest=True,
    )
    return evecs


def _lobpcg_initial_guess(
    N: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    X = torch.randn(N, k, dtype=dtype, generator=generator)
    if device.type != "cpu":
        X = X.to(device=device)

    return X
