# GPR_grad

Gaussian Process Regression with function-value and gradient observations, implemented in PyTorch.

`GPR_grad` is a compact research-oriented package for exact Gaussian process regression when the training data can include both scalar function values and derivative components. It also includes a deterministic CUR-style selector that can choose informative points from posterior covariance matrices.

## What is included

- `GradientGP`: exact GP regression with dense covariance matrices and Cholesky solves.
- `KernelFunction`: a kernel interface for function, function-gradient, and gradient-gradient covariance terms.
- `RBFKernelFunction`: an RBF-style squared exponential kernel with scalar or per-dimension length scales, plus vectorized block assembly.
- `log_marginal_likelihood`: differentiable GP marginal likelihood for optimizing kernel hyperparameters.
- `CUR_deterministic`: deterministic leverage-score-inspired column/row selection with exact and LOBPCG eigensolver options.
- `test/test-GP.ipynb`: a worked notebook showing noisy partial-gradient fitting, hyperparameter optimization, prediction, uncertainty estimates, and CUR selection.

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
gp = GradientGP(kernel=kernel, sigma_f=1e-6, sigma_g=1e-6, jitter=1e-8)

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

When `grad_indices` is provided, `G_g` may also be the full gradient array with shape `(Ng, D)`; the selected components are gathered internally. If `G_g` contains selected components only, `grad_indices` is required. Omitting it will raise a validation error instead of building a mismatched kernel and target vector.

## Hyperparameter optimization

`GradientGP.log_marginal_likelihood()` returns the differentiable log marginal likelihood for the most recent fit. This can be maximized to tune trainable kernel parameters such as RBF length scales.

```python
kernel = RBFKernelFunction(
    theta=torch.tensor([0.5, 0.5]),
    trainable=True,
)

gp = GradientGP(
    kernel=kernel,
    sigma_f=0.02,
    sigma_g=0.05,
    jitter=1e-8,
)

optimizer = torch.optim.Adam(gp.parameters(), lr=1e-2)

for step in range(300):
    optimizer.zero_grad()

    gp.fit(
        X_f=X_f,
        Y_f=Y_f,
        X_g=X_g,
        G_g=G_obs,
        grad_indices=grad_indices,
    )

    loss = -gp.log_marginal_likelihood()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(
            f"step={step:03d}",
            f"lml={-loss.item():.6f}",
            f"theta={kernel.theta.detach().tolist()}",
        )
```

`RBFKernelFunction(trainable=True)` stores `raw_theta` as the optimized parameter and exposes `theta = exp(raw_theta)`, so length scales stay positive without manual clamping.

Function and gradient observation noise can also be registered as parameters:

```python
gp = GradientGP(
    kernel=kernel,
    sigma_f=0.02,
    sigma_g=0.05,
    trainable_sigma_f=True,
    trainable_sigma_g=True,
)
```

The current noise parameters are optimized directly, so keep them positive after each optimizer step or replace them with your own log-parameterization if you need unconstrained optimization:

```python
with torch.no_grad():
    gp.sigma_f.clamp_(min=1e-8)
    gp.sigma_g.clamp_(min=1e-8)
```

## API notes

### `GradientGP`

```python
GradientGP(
    kernel,
    sigma_f=1e-6,
    sigma_g=1e-6,
    trainable_sigma_f=False,
    trainable_sigma_g=False,
    jitter=1e-8,
)
```

- `kernel`: a kernel module implementing the `KernelFunction` methods.
- `sigma_f`: observation noise standard deviation for function values.
- `sigma_g`: observation noise standard deviation for gradient values.
- `trainable_sigma_f`: whether to register `sigma_f` as a trainable parameter.
- `trainable_sigma_g`: whether to register `sigma_g` as a trainable parameter.
- `jitter`: diagonal stabilization added before Cholesky factorization.

Main methods:

- `fit(X_f, Y_f, X_g=None, G_g=None, grad_indices=None)`: stores training data, builds the joint function/gradient covariance matrix, and computes the Cholesky factorization.
- `log_marginal_likelihood()`: returns the differentiable log marginal likelihood for the most recent fit.
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

`theta` may be a scalar length scale or a tensor of per-dimension length scales. It must be positive. Internally, the kernel stores `raw_theta = log(theta)` and exposes `theta = exp(raw_theta)`. If `trainable=True`, `raw_theta` is a `torch.nn.Parameter`; otherwise it is registered as a module buffer.

The RBF kernel also provides vectorized block methods used automatically by `GradientGP`:

- `get_k_matrix(X, Y)`
- `get_k_fg_matrix(X, Y, grad_dims=None)`
- `get_k_gf_matrix(X, Y, grad_dims=None)`
- `get_k_gg_matrix(X, Y, x_grad_dims=None, y_grad_dims=None)`

Custom kernels do not need to implement these methods; `GradientGP` falls back to the scalar `KernelFunction` interface when vectorized methods are unavailable.

### `CUR_deterministic`

```python
rsel, rerror = CUR_deterministic(
    X,
    n_col,
    error_estimate=True,
    costs=1.0,
    eigensolver="auto",
)
```

- `X`: square covariance or similarity matrix with shape `(N, N)`.
- `n_col`: number of indices to select.
- `error_estimate`: whether to record residual error after each selection.
- `costs`: scalar or length-`N` tensor of selection costs.
- `eigensolver`: `"exact"` reproduces the original full eigendecomposition path, `"lobpcg"` uses an iterative top-k solve, and `"auto"` uses exact solves for small or unsuitable matrices and LOBPCG for larger top-k cases.

The function returns selected indices and residual-error estimates. This is useful for selecting high-uncertainty or high-leverage candidate points from a GP posterior covariance matrix.

## Running the notebook

From the repository root:

```bash
jupyter notebook test/test-GP.ipynb
```

The notebook demonstrates:

- adding random observation noise to function and gradient data,
- fitting with partial gradient components,
- optimizing RBF length scales with log marginal likelihood,
- checking interpolation at observed function and gradient data,
- predicting on a 2D grid,
- computing posterior function and gradient covariance,
- selecting candidate points with `CUR_deterministic`,
- visualizing predictions, errors, and uncertainty.

## Performance and limitations

`GradientGP` solves the dense exact GP problem with Cholesky factorization. The built-in RBF kernel uses vectorized block assembly for the function, function-gradient, and gradient-gradient covariance terms; custom kernels without vectorized block methods fall back to scalar Python loops.

Let `Nf` be the number of function observations and `Mg` be the number of observed derivative components. The joint training matrix has shape `(Nf + Mg, Nf + Mg)`, so exact inference requires dense linear algebra on that full matrix.

Keep tensors and modules on the same dtype and device. For example, if using CUDA tensors, move the model and kernel to CUDA as well.
