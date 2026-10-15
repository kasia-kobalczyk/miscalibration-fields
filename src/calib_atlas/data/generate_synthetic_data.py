import torch
from datasets import Dataset
import yaml
import numpy as np


def sigmoid(x):
    return torch.sigmoid(x)


def logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


def make_cluster_miscalibration_dataset(
    n: int = 10_000,
    K: int = 4,
    mu: np.ndarray | None = None,
    sigma_x: float = 0.5,
    w: np.ndarray | None = None,
    sigma_l: float = 0.5,
    delta: np.ndarray | None = None,
    s: np.ndarray | None = None,
    seed: int = 0,
    d: int = 2,
):
    """
    Cluster-structured miscalibration dataset with:
      - X | C=c ~ N(mu_c, sigma_x^2 I)
      - f(x)=sigmoid(l(x)), where l(x)= w^T (x - mu_c) + eps
        => f's distribution is (approximately) identical across clusters by construction
      - eta(x)=sigmoid(l(x) + s_c * delta_c) => cluster-dependent miscalibration sign & strength
      - y ~ Bernoulli(eta)

    Observed: (x_i, f_i, y_i), residual r_i = y_i - f_i.

    Returns dict with keys: x, f, y, r; and optionally c, eta, l, mu, w, delta, s.
    """
    rng = np.random.default_rng(seed)
    if mu is None:
        mu = rng.normal(0, 3.0, size=(K, d))
    mu = np.asarray(mu, dtype=float)
    assert mu.shape[0] == K, f"mu must have shape (K,d); got {mu.shape}"

    d = mu.shape[1]

    # Shared direction for logit on centered features
    if w is None:
        w = rng.normal(0, 1.0, size=(d,))
    w = np.asarray(w, dtype=float)
    w = w / (np.linalg.norm(w) + 1e-12)

    # Cluster-specific signs and magnitudes
    if s is None:
        s = np.ones(K, dtype=float)
        s[K // 2 :] = -1.0
    s = np.asarray(s, dtype=float)
    assert s.shape == (K,)

    if delta is None:
        base = np.array([0.2, 0.6], dtype=float)  # alternating strengths
        delta = np.tile(base, int(np.ceil(K / 2)))[:K]
    delta = np.asarray(delta, dtype=float)
    assert delta.shape == (K,)
    assert np.all(delta >= 0)

    # Sample clusters uniformly
    c = rng.integers(0, K, size=n)

    # Sample inputs
    x = mu[c] + sigma_x * rng.normal(0, 1.0, size=(n, d))

    # Latent logit depends on x but is cluster-invariant in distribution (via centering)
    centered = x - mu[c]
    l = centered @ w + sigma_l * rng.normal(0, 1.0, size=n)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    f = sigmoid(l)

    # Cluster-dependent miscalibration (sign and strength)
    eta = sigmoid(l + s[c] * delta[c])

    # Labels
    y = rng.binomial(1, eta).astype(int)

    # Delta
    delta = s[c] * delta[c]

    out = {
        "emb": torch.tensor(x.astype(np.float32)),
        "f": torch.tensor(f.astype(np.float32)),
        "y": torch.tensor(y.astype(np.int8)),
        "eta": torch.tensor(eta.astype(np.float32)),
        "delta": torch.tensor(delta.astype(np.float32)),
        "c": torch.tensor(c.astype(np.int8)),
    }

    if K == 2:
        g = np.where(s[c] >= 0.0, 1.0, 0.0).astype(np.int8)
        out.update({"g": torch.tensor(g.astype(np.int8))})

    return out


def sample_dense_sparse_blobs(
    n: int,
    w_dense: float = 0.8,
    mu_dense=None,
    mu_sparse=None,
    sigma_dense: float = 0.2,
    sigma_sparse: float = 1.0,
    d: int = 2,
    seed: int = 0,
    return_component: bool = True,
):
    """
    Sample X from a 2-component isotropic Gaussian mixture:

        X ~ w_dense * N(mu_dense, sigma_dense^2 I)
          + (1-w_dense) * N(mu_sparse, sigma_sparse^2 I)

    Parameters
    ----------
    n : int
        Number of samples.
    w_dense : float
        Mixture weight for dense component (probability of drawing from dense).
    mu_dense, mu_sparse : array-like shape (d,), optional
        Means of the dense/sparse components. If None, defaults to (-1,0,..) and (+1,0,..).
    sigma_dense, sigma_sparse : float
        Standard deviations of the dense/sparse components (sigma_dense < sigma_sparse).
    d : int
        Dimensionality.
    seed : int
        RNG seed.
    return_component : bool
        If True, return component id c in {0=dense, 1=sparse}.

    Returns
    -------
    x : (n,d) ndarray
    c : (n,) ndarray of ints, optional
    """
    rng = np.random.default_rng(seed)
    if not (0.0 <= w_dense <= 1.0):
        raise ValueError("w_dense must be in [0,1].")

    if mu_dense is None:
        mu_dense = np.zeros(d, dtype=float)
        mu_dense[0] = -1.0
    if mu_sparse is None:
        mu_sparse = np.zeros(d, dtype=float)
        mu_sparse[0] = 1.0

    mu_dense = np.asarray(mu_dense, dtype=float).reshape(d)
    mu_sparse = np.asarray(mu_sparse, dtype=float).reshape(d)

    # Component indicator: 0=dense w.p. w_dense, 1=sparse otherwise
    c = (rng.random(n) >= w_dense).astype(int)

    x = np.empty((n, d), dtype=float)
    idx0 = np.where(c == 0)[0]
    idx1 = np.where(c == 1)[0]

    if idx0.size:
        x[idx0] = mu_dense + sigma_dense * rng.normal(size=(idx0.size, d))
    if idx1.size:
        x[idx1] = mu_sparse + sigma_sparse * rng.normal(size=(idx1.size, d))

    return (x, c) if return_component else x


def make_smooth_field_miscalibration_dataset(
    n: int = 10_000,
    domain: float = 1.0,
    domain_type: str = "uniform_square",
    w: np.ndarray | None = None,
    sigma_l: float = 0.7,
    field_type: str = "sinusoid",  # "sinusoid" or "linear" or "rbf"
    A: float = 0.1,  # amplitude of miscalibration field (logit shift)
    u: np.ndarray | None = None,  # direction for sinusoid/linear
    freq: float = 1.0,  # sinusoid frequency multiplier
    rbf_centers: np.ndarray | None = None,
    rbf_scales: np.ndarray | None = None,
    seed: int = 0,
    return_latent: bool = True,
):
    """
    Smooth-field synthetic dataset where miscalibration varies continuously over x.

    Generate:
      X ~ Uniform([-domain, domain]^2)
      l(x) = w^T x + eps, eps ~ N(0, sigma_l^2)
      f(x) = sigmoid(l(x))
      Delta(x) = smooth logit-shift field (sinusoid/linear/RBF mixture)
      eta(x) = sigmoid(l(x) + Delta(x))
      Y ~ Bernoulli(eta(x))
      residual r = y - f

    Observed triples: (x, f, y).
    """
    rng = np.random.default_rng(seed)

    # Sample X
    if domain_type == "uniform_square":
        x = rng.uniform(-domain, domain, size=(n, 2))
    elif domain_type == "gaussian_blobs":
        x, _ = sample_dense_sparse_blobs(
            n, w_dense=0.7, sigma_dense=0.4, sigma_sparse=1.0, seed=seed
        )

    # Define w and u (unit vectors)
    if w is None:
        w = rng.normal(0, 1.0, size=(2,))
    w = np.asarray(w, dtype=float)
    w = w / (np.linalg.norm(w) + 1e-12)

    if u is None:
        u = rng.normal(0, 1.0, size=(2,))
    u = np.asarray(u, dtype=float)
    u = u / (np.linalg.norm(u) + 1e-12)

    # Latent logit for score
    eps = sigma_l * rng.normal(0, 1.0, size=n)
    l = x @ w + eps

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    f = sigmoid(l)

    # Smooth field Delta(x) in logit space
    ft = field_type.lower()
    if ft == "sinusoid":
        # u^T x ranges in [-sqrt(2)domain, +sqrt(2)domain]
        phase = 2.0 * np.pi * freq * (x @ u) / (2.0 * domain * np.sqrt(2.0))
        Delta = A * np.sin(phase)
    elif ft == "linear":
        Delta = A * (x @ u) / (domain * np.sqrt(2.0))
    elif ft == "rbf":
        if rbf_centers is None:
            rbf_centers = np.array([[-0.5, -0.5], [0.5, -0.25], [0.0, 0.6]]) * domain
        rbf_centers = np.asarray(rbf_centers, dtype=float)
        M = rbf_centers.shape[0]
        if rbf_scales is None:
            rbf_scales = np.full(M, 0.5 * domain)
        rbf_scales = np.asarray(rbf_scales, dtype=float)
        assert rbf_scales.shape == (M,)

        # signed weights with zero mean (encourage cancellation)
        weights = rng.normal(0, 1.0, size=M)
        weights = weights - weights.mean()
        weights = weights / (np.linalg.norm(weights) + 1e-12)

        diffs = x[:, None, :] - rbf_centers[None, :, :]
        sq = np.sum(diffs**2, axis=2)
        Phi = np.exp(-0.5 * sq / (rbf_scales[None, :] ** 2))
        Delta_raw = Phi @ weights
        Delta = A * Delta_raw / (np.max(np.abs(Delta_raw)) + 1e-12)
    else:
        raise ValueError("field_type must be one of {'sinusoid','linear','rbf'}")

    # Ground-truth conditional and labels
    eta = sigmoid(l + Delta)
    y = rng.binomial(1, eta).astype(int)
    r = y - f

    out = {"emb": x, "f": f, "y": y, "r": r}
    if return_latent:
        out.update(
            {
                "l": l,
                "eta": eta,
                "delta": Delta,
            }
        )

    for k, v in out.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.tensor(v.astype(np.float32))

    return out


def _save(data: dict, out_path: str, configs: dict) -> None:
    dataset = Dataset.from_dict({k: v.cpu().numpy() for k, v in data.items()})
    dataset.save_to_disk(out_path)
    with open(f"{out_path}/generation_config.yaml", "w") as f:
        yaml.dump(configs, f)
    print("Wrote", out_path, "with columns:", dataset.column_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic miscalibration datasets.")
    parser.add_argument("kind", choices=["clusters", "sinusoid"])
    parser.add_argument("--out_dir", default="./datasets")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--A", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help="sinusoid: amplitude(s) of the logit-shift field")
    parser.add_argument("--freq", type=float, nargs="+", default=[1, 2, 3, 4, 5],
                        help="sinusoid: frequency multiplier(s)")
    args = parser.parse_args()

    if args.kind == "clusters":
        # Three Gaussian clusters: over-, well- and under-confident by construction.
        configs = {
            "n": args.n,
            "K": 3,
            "mu": np.array([[-2.0, 1.0], [0.0, 0.0], [2.0, 1.0]]),
            "delta": np.array([1.0, 0.0, 1.0]),
            "sigma_l": 1.0,
            "seed": args.seed,
        }
        data = make_cluster_miscalibration_dataset(**configs)
        _save(data, f"{args.out_dir}/synthetic_3_clusters",
              {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in configs.items()})
    else:
        for freq in args.freq:
            for A in args.A:
                freq = int(freq) if float(freq).is_integer() else freq
                configs = {"n": args.n, "field_type": "sinusoid", "A": A, "freq": freq, "seed": args.seed}
                data = make_smooth_field_miscalibration_dataset(**configs)
                _save(data, f"{args.out_dir}/synthetic_field_sinusoid_A{A}_freq{freq}", configs)
