import torch
import torch.nn as nn

__all__ = ["GradientGP"]

class GradientGP(nn.Module):
    def __init__(
        self,
        kernel: nn.Module,
        sigma_f: float = 1e-6,
        sigma_g: float = 1e-6,
        jitter: float = 1e-8,
    ):
        super().__init__()
        self.kernel = kernel
        self.sigma_f = sigma_f
        self.sigma_g = sigma_g
        self.jitter = jitter

    # =========================
    # normalize gradient inputs
    # =========================
    def _normalize_grad_inputs(self, X_g, G_g=None, grad_indices=None):
        if X_g is None:
            return None, None, None

        Ng, D = X_g.shape

        if grad_indices is None:
            # full gradients
            point_ids = torch.arange(Ng, device=X_g.device).repeat_interleave(D)
            dim_ids = torch.arange(D, device=X_g.device).repeat(Ng)
            grad_indices = torch.stack([point_ids, dim_ids], dim=1)

            G_obs = G_g.reshape(-1) if G_g is not None else None
        else:
            grad_indices = grad_indices.to(X_g.device)
            point_ids = grad_indices[:, 0]
            dim_ids = grad_indices[:, 1]
            G_obs = G_g.reshape(-1) if G_g is not None else None

        X_grad_obs = X_g[point_ids]
        grad_dims = dim_ids

        return X_grad_obs, grad_dims, G_obs

    # =========================
    # training kernel
    # =========================
    def build_kernel_matrix(self, X_f, X_grad_obs, grad_dims):
        Nf = X_f.shape[0]
        Mg = 0 if X_grad_obs is None else X_grad_obs.shape[0]

        dtype = X_f.dtype
        device = X_f.device

        K_ff = torch.empty(Nf, Nf, dtype=dtype, device=device)

        for i in range(Nf):
            for j in range(Nf):
                K_ff[i, j] = self.kernel.get_k(X_f[i], X_f[j])

        if Mg == 0:
            return K_ff

        K_fg = torch.empty(Nf, Mg, dtype=dtype, device=device)
        K_gf = torch.empty(Mg, Nf, dtype=dtype, device=device)
        K_gg = torch.empty(Mg, Mg, dtype=dtype, device=device)

        for i in range(Nf):
            for m in range(Mg):
                K_fg[i, m] = self.kernel.get_k_fg(
                    X_f[i], X_grad_obs[m], dim=grad_dims[m]
                )

        for m in range(Mg):
            for j in range(Nf):
                K_gf[m, j] = self.kernel.get_k_gf(
                    X_grad_obs[m], X_f[j], dim=grad_dims[m]
                )

        for m in range(Mg):
            for n in range(Mg):
                K_gg[m, n] = self.kernel.get_k_gg(
                    X_grad_obs[m],
                    X_grad_obs[n],
                    dims=[grad_dims[m], grad_dims[n]],
                )

        return torch.cat([
            torch.cat([K_ff, K_fg], dim=1),
            torch.cat([K_gf, K_gg], dim=1),
        ], dim=0)

    # =========================
    # cross kernel
    # =========================
    def build_cross_kernel(self, X_star):
        Ns, D = X_star.shape
        Nf = self.X_f.shape[0]
        Mg = 0 if self.X_grad_obs is None else self.X_grad_obs.shape[0]

        K_sf = torch.empty(Ns, Nf, dtype=X_star.dtype, device=X_star.device)

        for i in range(Ns):
            for j in range(Nf):
                K_sf[i, j] = self.kernel.get_k(X_star[i], self.X_f[j])

        if Mg == 0:
            return K_sf

        K_sg = torch.empty(Ns, Mg, dtype=X_star.dtype, device=X_star.device)

        for i in range(Ns):
            for m in range(Mg):
                K_sg[i, m] = self.kernel.get_k_fg(
                    X_star[i],
                    self.X_grad_obs[m],
                    dim=self.grad_dims[m],
                )

        return torch.cat([K_sf, K_sg], dim=1)

    # =========================
    # noise
    # =========================
    def build_noise_matrix(self, Nf, Mg, dtype, device):
        noise_f = self.sigma_f**2 * torch.ones(Nf, dtype=dtype, device=device)
        noise_g = self.sigma_g**2 * torch.ones(Mg, dtype=dtype, device=device)
        return torch.diag(torch.cat([noise_f, noise_g]))

    # =========================
    # fit
    # =========================
    def fit(self, X_f, Y_f, X_g=None, G_g=None, grad_indices=None):
        self.X_f = X_f
        self.Y_f = Y_f.reshape(-1)

        X_grad_obs, grad_dims, G_obs = self._normalize_grad_inputs(
            X_g, G_g, grad_indices
        )

        self.X_grad_obs = X_grad_obs
        self.grad_dims = grad_dims

        if G_obs is not None:
            y_train = torch.cat([self.Y_f, G_obs])
        else:
            y_train = self.Y_f

        Mg = 0 if X_grad_obs is None else X_grad_obs.shape[0]

        K_xx = self.build_kernel_matrix(X_f, X_grad_obs, grad_dims)
        Sigma = self.build_noise_matrix(
            X_f.shape[0], Mg, dtype=K_xx.dtype, device=K_xx.device
        )

        K = K_xx + Sigma
        K = K + self.jitter * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)

        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(y_train[:, None], self.L).squeeze(-1)

        return self

    # =========================
    # predict
    # =========================
    def predict(self, X_star, return_cov=True):
        K_sx = self.build_cross_kernel(X_star)
        mean = K_sx @ self.alpha

        if not return_cov:
            return mean

        Ns = X_star.shape[0]

        K_ss = torch.empty(Ns, Ns, dtype=X_star.dtype, device=X_star.device)

        for i in range(Ns):
            for j in range(Ns):
                K_ss[i, j] = self.kernel.get_k(X_star[i], X_star[j])

        v = torch.cholesky_solve(K_sx.T, self.L)
        cov = K_ss - K_sx @ v

        return mean, cov

    # =========================
    # gradient covariance
    # =========================
    def build_prior_gradient_kernel(self, X_star):
        Ns, D = X_star.shape
        K = torch.empty(Ns * D, Ns * D, dtype=X_star.dtype, device=X_star.device)

        for i in range(Ns):
            for a in range(D):
                row = i * D + a
                for j in range(Ns):
                    for b in range(D):
                        col = j * D + b
                        K[row, col] = self.kernel.get_k_gg(
                            X_star[i], X_star[j], dims=[a, b]
                        )
        return K

    def build_cross_kernel_gradients(self, X_star):
        Ns, D = X_star.shape
        Nf = self.X_f.shape[0]
        Mg = 0 if self.X_grad_obs is None else self.X_grad_obs.shape[0]

        K_gs_f = torch.empty(Ns * D, Nf, dtype=X_star.dtype, device=X_star.device)

        for i in range(Ns):
            for a in range(D):
                row = i * D + a
                for j in range(Nf):
                    K_gs_f[row, j] = self.kernel.get_k_gf(
                        X_star[i], self.X_f[j], dim=a
                    )

        if Mg == 0:
            return K_gs_f

        K_gs_g = torch.empty(Ns * D, Mg, dtype=X_star.dtype, device=X_star.device)

        for i in range(Ns):
            for a in range(D):
                row = i * D + a
                for m in range(Mg):
                    K_gs_g[row, m] = self.kernel.get_k_gg(
                        X_star[i],
                        self.X_grad_obs[m],
                        dims=[a, self.grad_dims[m]],
                    )

        return torch.cat([K_gs_f, K_gs_g], dim=1)

    def predict_grad_cov(self, X_star):
        K_ss = self.build_prior_gradient_kernel(X_star)
        K_sx = self.build_cross_kernel_gradients(X_star)

        v = torch.cholesky_solve(K_sx.T, self.L)
        cov_grad = K_ss - K_sx @ v

        return 0.5 * (cov_grad + cov_grad.T)
    
    
    def predict_grad_cov_by_points(
        self,
        X_star,
        grad_indices: torch.Tensor = None,
    ):
        """
        Convert gradient covariance to point-level covariance.

        Case 1: full gradients
            cov_grad: (Ns*D, Ns*D)
            returns:  (Ns, Ns)
            uses trace of each D x D block

        Case 2: partial gradients
            cov_grad:     (Mg, Mg)
            grad_indices: (Mg, 2), rows = [point_id, grad_dim]
            returns:      (Ns, Ns)
            sums matching derivative-component covariances by point_id
        """
        Ns, D = X_star.shape
        cov_grad = self.predict_grad_cov(X_star)
        
        if grad_indices is None:
            
            K_point = torch.empty(
                Ns, Ns, dtype=cov_grad.dtype, device=cov_grad.device
            )

            for i in range(Ns):
                for j in range(Ns):
                    block = cov_grad[i*D:(i+1)*D, j*D:(j+1)*D]
                    K_point[i, j] = torch.trace(block)

            return K_point

        else:
            grad_indices = grad_indices.to(cov_grad.device)
            point_ids = grad_indices[:, 0]

            K_point = torch.zeros(
                Ns, Ns, dtype=cov_grad.dtype, device=cov_grad.device
            )

            Mg = grad_indices.shape[0]

            for m in range(Mg):
                i = point_ids[m].item()
                for n in range(Mg):
                    j = point_ids[n].item()
                    K_point[i, j] += cov_grad[m, n]

            return K_point
