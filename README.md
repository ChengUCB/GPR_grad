# GPR_grad

Gaussian Process Regression with function-value and gradient observations, implemented in PyTorch.

`GPR_grad` is a compact research-oriented package for exact Gaussian process regression when the training data can include both scalar function values and derivative components. It also includes a deterministic CUR-style selector that can choose informative points from posterior covariance matrices.

## What is included

- `GradientGP`: exact GP regression with dense covariance matrices and Cholesky solves.
- `KernelFunction`: a kernel interface for function, function-gradient, and gradient-gradient covariance terms.
- `RBFKernelFunction`: an RBF-style squared exponential kernel with scalar or per-dimension length scales.
- `CUR_deterministic`: deterministic leverage-score-inspired column/row selection for covariance or similarity matrices.
- `test/test-GP.ipynb`: a worked notebook showing partial-gradient fitting, prediction, uncertainty estimates, and CUR selection.

## Repository layout

```text
.
|-- README.md
|-- gpr_grad
|   |-- __init__.py
|   |-- cur.py
|   |-- gpr.py
|   `-- kernel_func.py
`-- test
    `-- test-GP.ipynb
```

## Requirements

- Python 3
- PyTorch
- Jupyter and Matplotlib, only for running the example notebook

This repository does not currently include packaging metadata such as `pyproject.toml` or `setup.py`, so use it from the repository root or add the repository root to `PYTHONPATH`.

## Quick start

```python
import torch
from gpr_grad import GradientGP, RBFKernelFunction, CUR_deterministic

torch.set_default_dtype(torch.float64)

def f_true(X):
    x = X[:, 0]
    y = X[:, 1]
    return torch.sin(2.0 * x) + 0.5 * torch.cos(3.0 * y)

def grad_true(X):
    x = X[:, 0]
    y = X[:, 1]
    return torch.stack(
        [
            2.0 * torch.cos(2.0 * x),
            -1.5 * torch.sin(3.0 * y),
        ],
        dim=1,
    )

X_f = torch.tensor(
    [
        [-1.0, -1.0],
        [0.0, 0.0],
    ]
)
Y_f = f_true(X_f)

X_g = torch.tensor(
    [
        [-0.8, 0.5],
        [0.3, -0.4],
        [0.9, 0.7],
    ]
)
G_g = grad_true(X_g)

kernel = RBFKernelFunction(theta=torch.tensor([0.8, 0.8]))
gp = GradientGP(kernel=kernel, sigma_f=1e-8, sigma_g=1e-8, jitter=1e-10)

gp.fit(X_f=X_f, Y_f=Y_f, X_g=X_g, G_g=G_g)

X_star = torch.tensor(
    [
        [-0.5, -0.5],
        [0.0, 0.0],
        [0.5, 0.5],
    ]
)

mean, cov = gp.predict(X_star)
grad_cov = gp.predict_grad_cov(X_star)
point_grad_cov = gp.predict_grad_cov_by_points(X_star)

selected, residual_error = CUR_deterministic(
    point_grad_cov.detach(),
    n_col=2,
    error_estimate=True,
)

print(mean)
print(selected)
```

## Partial-gradient observations

`GradientGP.fit` can use either full gradients at every gradient-observation point or selected derivative components.

For full gradients, pass `X_g` with shape `(Ng, D)` and `G_g` with shape `(Ng, D)`:

```python
gp.fit(X_f=X_f, Y_f=Y_f, X_g=X_g, G_g=G_full)
```

For partial gradients, pass `grad_indices` with shape `(Mg, 2)`. Each row is `[point_index, dimension_index]`, where `point_index` indexes into `X_g` and `dimension_index` selects the derivative component.

```python
grad_indices = torch.tensor(
    [
        [0, 0],  # df/dx at X_g[0]
        [1, 1],  # df/dy at X_g[1]
        [2, 0],  # df/dx at X_g[2]
    ],
    dtype=torch.long,
)

G_obs = torch.stack(
    [
        G_full[point_id, dim_id]
        for point_id, dim_id in grad_indices
    ]
)

gp.fit(
    X_f=X_f,
    Y_f=Y_f,
    X_g=X_g,
    G_g=G_obs,
    grad_indices=grad_indices,
)
```

## API notes

### `GradientGP`

```python
GradientGP(kernel, sigma_f=1e-6, sigma_g=1e-6, jitter=1e-8)
```

- `kernel`: a kernel module implementing the `KernelFunction` methods.
- `sigma_f`: observation noise standard deviation for function values.
- `sigma_g`: observation noise standard deviation for gradient values.
- `jitter`: diagonal stabilization added before Cholesky factorization.

Main methods:

- `fit(X_f, Y_f, X_g=None, G_g=None, grad_indices=None)`: stores training data, builds the joint function/gradient covariance matrix, and computes the Cholesky factorization.
- `predict(X_star, return_cov=True)`: returns posterior function mean and, optionally, posterior function covariance.
- `predict_grad_cov(X_star)`: returns posterior covariance over all derivative components at `X_star`, with shape `(Ns * D, Ns * D)`.
- `predict_grad_cov_by_points(X_star, grad_indices=None)`: converts derivative-component covariance into a point-level covariance matrix with shape `(Ns, Ns)`.

### `KernelFunction`

Custom kernels should implement:

- `get_k(x, y)`: covariance between function values.
- `get_k_fg(x, y, dim=None)`: derivative of `k(x, y)` with respect to `y`.
- `get_k_gg(x, y, dims=None)`: mixed second derivative covariance.

`get_k_gf(x, y, dim=None)` defaults to `-get_k_fg(x, y)` and can be overridden when needed.

### `RBFKernelFunction`

```python
kernel = RBFKernelFunction(theta=torch.tensor([0.8, 0.8]), trainable=False)
```

The built-in kernel computes:

```text
k(x, y) = exp(-0.5 * sum(((x - y) / theta) ** 2))
```

`theta` may be a scalar length scale or a tensor of per-dimension length scales. If `trainable=True`, `theta` is stored as a `torch.nn.Parameter`; otherwise it is registered as a module buffer.

### `CUR_deterministic`

```python
rsel, rerror = CUR_deterministic(X, n_col, error_estimate=True, costs=1.0)
```

- `X`: square covariance or similarity matrix with shape `(N, N)`.
- `n_col`: number of indices to select.
- `error_estimate`: whether to record residual error after each selection.
- `costs`: scalar or length-`N` tensor of selection costs.

The function returns selected indices and residual-error estimates. This is useful for selecting high-uncertainty or high-leverage candidate points from a GP posterior covariance matrix.

## Running the notebook

From the repository root:

```bash
jupyter notebook test/test-GP.ipynb
```

The notebook demonstrates:

- fitting with partial gradient components,
- checking interpolation at observed function and gradient data,
- predicting on a 2D grid,
- computing posterior function and gradient covariance,
- selecting candidate points with `CUR_deterministic`,
- visualizing predictions, errors, and uncertainty.

## Performance and limitations

`GradientGP` builds dense covariance matrices using Python loops and solves them exactly with Cholesky factorization. This makes the implementation straightforward and useful for small experiments, but memory and runtime scale poorly for large datasets.

Let `Nf` be the number of function observations and `Mg` be the number of observed derivative components. The joint training matrix has shape `(Nf + Mg, Nf + Mg)`, so exact inference requires dense linear algebra on that full matrix.

Keep tensors and modules on the same dtype and device. For example, if using CUDA tensors, move the model and kernel to CUDA as well.
