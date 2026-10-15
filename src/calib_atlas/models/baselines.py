import numpy as np
import torch
import torch.nn.functional as F

from calib_atlas.utils.eval_utils import compute_hat_delta_from_bank


class TemperatureScaler:
    """
    Temperature scaling fit by full-batch L-BFGS on binary NLL.
    Input/output are 1D probabilities in (0, 1).
    """

    def __init__(
        self,
        lr: float = 0.1,
        max_iter: int = 50,
        eps: float = 1e-6,
        **_legacy_kwargs,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self._best_T: float | None = None

    @staticmethod
    def _to_1d(x, dtype=torch.float64) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype).flatten()

    def _logits(self, f: torch.Tensor) -> torch.Tensor:
        f = f.clamp(self.eps, 1 - self.eps)
        return torch.log(f) - torch.log1p(-f)

    def fit(self, f_train, y_train, f_val=None, y_val=None, seed: int = 0):
        torch.manual_seed(seed)

        f = self._to_1d(f_train)
        y = self._to_1d(y_train)
        z = self._logits(f)

        log_T = torch.zeros((), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS(
            [log_T],
            lr=self.lr,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(z / log_T.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        self._best_T = float(log_T.exp().detach())
        return self

    @property
    def T_(self) -> float:
        if self._best_T is None:
            raise RuntimeError("TemperatureScaler is not fit yet. Call fit(...) first.")
        return self._best_T

    def transform(self, f):
        T = self.T_
        f = self._to_1d(f, dtype=torch.float32)
        z = self._logits(f)
        return torch.sigmoid(z / T).cpu().numpy()

    def predict_proba(self, f):
        return self.transform(f)

    def get_params(self, deep: bool = True):
        return {"lr": self.lr, "max_iter": self.max_iter, "eps": self.eps}

    def __repr__(self):
        if self._best_T is None:
            return f"TemperatureScaler(unfit, lr={self.lr}, max_iter={self.max_iter})"
        return f"TemperatureScaler(T_={self._best_T:.6f}, lr={self.lr}, max_iter={self.max_iter})"


class RawEmbeddingKernelSmoother:
    """
    Nadaraya-Watson kernel smoothing of residuals on raw LLM embeddings.

    Baseline counterpart to the proposed method's learned-phi kernel: instead
    of smoothing residuals over the learned z = phi(x), this smooths over the
    raw embedding x (optionally L2-normalized) directly. Reuses the existing
    `compute_hat_delta_from_bank` so the kernel arithmetic is identical.

    Sklearn-style API: `fit(x_train, f_train, y_train).predict_hat_delta(x_q, f_q)`.
    """

    def __init__(
        self,
        sigma: float,
        tau: float | None = None,
        normalize: bool = True,
        eps: float = 1e-8,
        chunk_q: int = 2048,
    ):
        self.sigma = sigma
        self.tau = tau
        self.normalize = normalize
        self.eps = eps
        self.chunk_q = chunk_q
        self._x_bank: torch.Tensor | None = None
        self._f_bank: torch.Tensor | None = None
        self._r_bank: torch.Tensor | None = None

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)

    def _prep(self, x) -> torch.Tensor:
        x = torch.as_tensor(x).float()
        if self.normalize:
            x = self._l2_normalize(x, self.eps)
        return x

    def fit(self, x_train, f_train, y_train):
        x = self._prep(x_train)
        f = torch.as_tensor(f_train).float().flatten()
        y = torch.as_tensor(y_train).float().flatten()
        assert x.shape[0] == f.shape[0] == y.shape[0]
        self._x_bank = x
        self._f_bank = f
        self._r_bank = y - f
        return self

    def predict_hat_delta(
        self,
        x_query,
        f_query,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        if self._x_bank is None:
            raise RuntimeError(
                "RawEmbeddingKernelSmoother is not fit yet. Call fit(...) first."
            )

        x_q = self._prep(x_query).to(self._x_bank.device)
        f_q = torch.as_tensor(f_query).float().flatten().to(self._x_bank.device)

        hat, _ = compute_hat_delta_from_bank(
            z_query=x_q,
            f_query=f_q,
            z_bank=self._x_bank,
            f_bank=self._f_bank,
            r_bank=self._r_bank,
            sigma=self.sigma,
            tau=self.tau,
            eps=self.eps,
            chunk_q=self.chunk_q,
            exclude_self=exclude_self,
        )
        return hat

    def __repr__(self):
        n_bank = 0 if self._x_bank is None else self._x_bank.shape[0]
        return (
            f"RawEmbeddingKernelSmoother(sigma={self.sigma}, tau={self.tau}, "
            f"normalize={self.normalize}, n_bank={n_bank})"
        )
