import torch
import torch.nn as nn
from torch.utils.data import Subset
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar
from calib_atlas.losses.kernel_calib import kernel_calib_discovery_loss
from calib_atlas.losses.baseline_calib import regression_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import relplot


@torch.no_grad()
def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.float()
    b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b) / (torch.sqrt((a @ a) + eps) * torch.sqrt((b @ b) + eps))


@torch.no_grad()
def auc_rank(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    AUC for binary labels {0,1} using rank statistic.
    Returns AUC in [0,1]. If all labels same, returns nan.
    """
    scores = scores.float()
    labels = labels.long()
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float("nan"), device=scores.device)

    # ranks of scores (average ranks for ties)
    sorted_scores, idx = torch.sort(scores)
    sorted_labels = labels[idx]

    # Compute average rank for ties
    # ranks are 1..N
    N = scores.numel()
    ranks = torch.empty(N, device=scores.device, dtype=torch.float32)
    i = 0
    rank = 1
    while i < N:
        j = i
        while j + 1 < N and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i)))
        ranks[i : j + 1] = avg_rank
        rank += j - i + 1
        i = j + 1

    # sum ranks of positives
    sum_ranks_pos = ranks[sorted_labels == 1].sum()
    # Mann–Whitney U
    U = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = U / (n_pos * n_neg)
    return auc


@torch.no_grad()
def compute_hat_delta_from_bank(
    z_query: torch.Tensor,
    f_query: torch.Tensor,
    z_bank: torch.Tensor,
    f_bank: torch.Tensor,
    r_bank: torch.Tensor,
    sigma: float,
    tau: float | None,
    eps: float = 1e-8,
    chunk_q: int = 2048,
    exclude_self: bool = False,
) -> torch.Tensor:
    """
    Compute hat_Delta for each query point using a fixed bank.
    Complexity: O(|Q|*|B|). Chunked to avoid OOM.

    z_* should already be normalized if you trained that way.
    """
    device = z_query.device
    Q = z_query.shape[0]
    hat = torch.empty(Q, device=device, dtype=torch.float32)
    emass = torch.empty(Q, device=device, dtype=torch.float32)

    for s in range(0, Q, chunk_q):
        e = min(Q, s + chunk_q)
        zq = z_query[s:e]  # (q,d)
        fq = f_query[s:e].view(-1, 1)  # (q,1)

        # squared distances between zq and z_bank
        # torch.cdist: (q,B)
        d2 = torch.cdist(zq, z_bank, p=2.0) ** 2
        Kx = torch.exp(-d2 / (sigma**2 + eps))

        if tau is not None:
            df2 = (fq - f_bank.view(1, -1)) ** 2
            Kf = torch.exp(-df2 / (tau**2 + eps))
            K = Kx * Kf
        else:
            K = Kx

        if exclude_self:
            assert z_query.shape == z_bank.shape and torch.allclose(z_query, z_bank)
            # Zero out diagonal entries (self-distances)
            K[torch.arange(K.shape[0]), s:e] = 0.0

        K = K.float()
        r_bank = r_bank.float()

        emass[s:e] = K.sum(dim=1) + eps  # (q,)
        num = (K @ r_bank.view(-1, 1)).view(-1)  # (q,)
        hat[s:e] = num / emass[s:e]

    return hat, emass


@torch.no_grad()
def build_bank(
    phi, loader, device: str, max_bank_size: int = 20000, model_type: str = "kernel"
):
    phi.eval()
    # Create a random subset of the loader of size max_bank_size
    N = min(len(loader.dataset), max_bank_size)
    idx = torch.randperm(len(loader.dataset))[:N]
    loader = torch.utils.data.DataLoader(
        Subset(loader.dataset, idx),
        batch_size=loader.batch_size,
        shuffle=True,
        collate_fn=loader.collate_fn,
        drop_last=False,
    )
    Zs, Fs, Rs = [], [], []
    for batch in loader:
        emb = batch["emb"].to(device)
        f = batch["f"].to(device).float()
        y = batch["y"].to(device).float()
        if model_type == "kernel_tokenwise":
            mask = batch["attention_mask"].to(device, non_blocking=True)
            emb = emb.squeeze(1)
            mask = mask.squeeze(1)
            z = phi(emb, mask=mask)
        else:
            z = phi(emb)
        r = y - f
        Zs.append(z)
        Fs.append(f)
        Rs.append(r)
    return torch.cat(Zs, dim=0), torch.cat(Fs, dim=0), torch.cat(Rs, dim=0)


@torch.no_grad()
def gather_data(phi, loader, cfg):
    phi.eval()
    device = cfg.device

    out = {}
    for batch in loader:
        emb = batch["emb"].to(device)
        if cfg.model.get("conditional", False):
            emb = torch.cat([emb, batch["f"].to(device).unsqueeze(1)], dim=1)
        if cfg.model.type == "kernel_tokenwise":
            mask = batch["attention_mask"].to(device, non_blocking=True)
            emb = emb.squeeze(1)
            mask = mask.squeeze(1)
            z = phi(emb, mask=mask)
        else:
            z = phi(emb)
        out["z"] = out.get("z", []) + [z]
        for k, v in batch.items():
            out[k] = out.get(k, []) + [
                v.to(device) if isinstance(v, torch.Tensor) else v
            ]

    for k in out:
        if isinstance(out[k][0], torch.Tensor):
            out[k] = torch.cat(out[k], dim=0)
        else:
            out[k] = [item for sublist in out[k] for item in sublist]

    if batch.get("g") is not None:
        out["has_g"] = True
    else:
        out["has_g"] = False

    if batch.get("eta") is not None:
        out["has_eta"] = True
    else:
        out["has_eta"] = False
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    eps = 1e-5
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_scale(f, y, hat_delta, correction="clip", objective="log_loss"):
    def _objective(scale):
        if correction == "clip":
            corrected_f = np.clip(f + scale * hat_delta, 0, 1)
        elif correction == "soft":
            corrected_f = sigmoid(logit(f) + scale * hat_delta)
        eps = 1e-5
        if objective == "log_loss":
            loss = -np.mean(
                y * np.log(corrected_f + eps) + (1 - y) * np.log(1 - corrected_f + eps)
            )
        elif objective == "brier_score":
            loss = ((corrected_f - y) ** 2).mean()
        else:
            raise ValueError(f"Unknown objective: {objective}")
        return loss

    res = minimize_scalar(_objective, bounds=(0.1, 10), method="bounded")
    return res.x


@torch.no_grad
def apply_soft_correction(f, hat_delta, scale):
    return sigmoid(logit(f) + scale * hat_delta)


@torch.no_grad
def best_delta_log_loss_soft(
    f: torch.Tensor,  # [N] predicted probabilities
    y: torch.Tensor,  # [N] labels in {0,1}
    hat_delta: torch.Tensor,  # [N] estimated local correction
    eps: float = 1e-6,
    reduction: str = "mean",  # "mean" or "sum"
) -> torch.Tensor:

    assert f.shape == y.shape == hat_delta.shape
    f = f.float()
    y = y.float()
    hat_delta = hat_delta.float()

    # Base probabilities
    f_base = f.clamp(eps, 1.0 - eps)

    # Locally corrected probabilities
    # Find the best scaling factor
    best_scale = fit_scale(
        f.cpu().numpy(), y.cpu().numpy(), hat_delta.cpu().numpy(), correction="soft"
    )

    f_local = nn.functional.sigmoid(torch.logit(f) + best_scale * hat_delta).clamp(
        eps, 1.0 - eps
    )

    # Log loss
    base_ll = -(y * torch.log(f_base) + (1.0 - y) * torch.log(1.0 - f_base))

    local_ll = -(y * torch.log(f_local) + (1.0 - y) * torch.log(1.0 - f_local))

    if reduction == "mean":
        base_ll = base_ll.mean()
        local_ll = local_ll.mean()
    elif reduction == "sum":
        base_ll = base_ll.sum()
        local_ll = local_ll.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    delta = base_ll - local_ll

    return delta.item()


@torch.no_grad()
def delta_log_loss(
    f: torch.Tensor,  # [N] predicted probabilities
    y: torch.Tensor,  # [N] labels in {0,1}
    hat_delta: torch.Tensor,  # [N] estimated local correction
    eps: float = 1e-6,
    reduction: str = "mean",  # "mean" or "sum"
) -> torch.Tensor:
    """
    Computes DeltaLogLoss = LogLoss(f) - LogLoss(f + hat_delta)

    Assumes hat_delta was computed using a TRAIN-ONLY bank.
    """

    assert f.shape == y.shape == hat_delta.shape
    f = f.float()
    y = y.float()
    hat_delta = hat_delta.float()

    # Base probabilities
    f_base = f.clamp(eps, 1.0 - eps)

    # Locally corrected probabilities
    f_local = (f + hat_delta).clamp(eps, 1.0 - eps)

    # Log loss
    base_ll = -(y * torch.log(f_base) + (1.0 - y) * torch.log(1.0 - f_base))

    local_ll = -(y * torch.log(f_local) + (1.0 - y) * torch.log(1.0 - f_local))

    if reduction == "mean":
        base_ll = base_ll.mean()
        local_ll = local_ll.mean()
    elif reduction == "sum":
        base_ll = base_ll.sum()
        local_ll = local_ll.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    delta = base_ll - local_ll

    return delta.item()


def shrunk_delta_log_loss(
    f: torch.Tensor,
    y: torch.Tensor,
    hat_delta: torch.Tensor,
    mass: torch.Tensor,
    lambda_: float,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    hat_delta_shrunk = mass / (mass + lambda_) * hat_delta
    return delta_log_loss(
        f=f,
        y=y,
        hat_delta=hat_delta_shrunk,
        eps=eps,
        reduction=reduction,
    )


@torch.no_grad()
def get_best_delta_log_loss(
    f, y, hat_delta, eps=1e-6, s_grid=None, shrunk=False, mass=None, lambda_=None
):
    if s_grid is None:
        s_grid = torch.linspace(-8, 0, 21, device=f.device)
        s_grid = 10**s_grid  # from 1e-8 to 1

    best_scale = fit_scale(
        f.cpu().numpy(), y.cpu().numpy(), hat_delta.cpu().numpy(), correction="clip"
    )

    best_dll = delta_log_loss(f, y, best_scale * hat_delta, eps=eps)

    return best_dll


@torch.no_grad()
def get_residual_mse(
    f: torch.Tensor, y: torch.Tensor, hat_delta: torch.Tensor
) -> torch.Tensor:
    r = f - y
    mse = (r - hat_delta) ** 2
    return mse.mean().item()


@torch.no_grad()
def get_corrected_bs_score(
    f: torch.Tensor,
    y: torch.Tensor,
    hat_delta: torch.Tensor,
    clip=False,
) -> torch.Tensor:
    corrected_f = f + hat_delta
    if clip:
        corrected_f = torch.clamp(corrected_f, 0, 1)
    bs = ((corrected_f - y) ** 2).mean().item()
    return bs


@torch.no_grad()
def get_best_corrected_bs_score(
    f: torch.Tensor,
    y: torch.Tensor,
    hat_delta: torch.Tensor,
) -> torch.Tensor:

    scale = fit_scale(
        f.cpu().numpy(),
        y.cpu().numpy(),
        hat_delta.cpu().numpy(),
        correction="clip",
        objective="brier_score",
    )
    corrected_f = torch.clamp(f + scale * hat_delta, 0, 1)
    bs = ((corrected_f - y) ** 2).mean().item()
    return bs


def range_aware_correct(
    f,
    hat_delta,
    alpha,
):
    """
    Smooth range-aware correction.

    For hat_delta < 0:
        corrected_f = f - f * tanh(alpha * |hat_delta|)

    For hat_delta >= 0:
        corrected_f = f + (1 - f) * tanh(alpha * |hat_delta|)

    This guarantees corrected_f in [0, 1] without clipping.
    """
    f = np.asarray(f)
    hat_delta = np.asarray(hat_delta)

    mag = np.tanh(alpha * np.abs(hat_delta))

    corrected_f = np.where(
        hat_delta < 0,
        f - f * mag,
        f + (1.0 - f) * mag,
    )

    return corrected_f


def fit_range_aware_scale(
    f,
    y,
    hat_delta,
    objective="brier_score",
):
    """
    Fit alpha > 0 for the range-aware correction map.
    """

    f = np.asarray(f)
    y = np.asarray(y)
    hat_delta = np.asarray(hat_delta)

    eps = 1e-8

    def _objective(alpha):
        corrected_f = range_aware_correct(
            f=f,
            hat_delta=hat_delta,
            alpha=alpha,
        )

        if objective == "brier_score":
            loss = np.mean((corrected_f - y) ** 2)

        elif objective == "log_loss":
            corrected_f = np.clip(corrected_f, eps, 1 - eps)
            loss = -np.mean(y * np.log(corrected_f) + (1 - y) * np.log(1 - corrected_f))

        else:
            raise ValueError(f"Unknown objective: {objective}")

        return loss

    res = minimize_scalar(
        _objective,
        bounds=(0.01, 50.0),
        method="bounded",
    )

    return res.x


@torch.no_grad()
def get_best_range_aware_bs_score(
    f: torch.Tensor,
    y: torch.Tensor,
    hat_delta: torch.Tensor,
):
    """
    Returns the best achievable corrected Brier score using the
    smooth range-aware correction map.
    """

    alpha = fit_range_aware_scale(
        f=f.cpu().numpy(),
        y=y.cpu().numpy(),
        hat_delta=hat_delta.cpu().numpy(),
        objective="brier_score",
    )

    f_np = f.cpu().numpy()
    hat_delta_np = hat_delta.cpu().numpy()

    mag = np.tanh(alpha * np.abs(hat_delta_np))

    corrected_f = np.where(
        hat_delta_np < 0,
        f_np - f_np * mag,
        f_np + (1.0 - f_np) * mag,
    )

    corrected_f = torch.tensor(corrected_f, device=f.device, dtype=f.dtype)

    bs = ((corrected_f - y) ** 2).mean().item()

    return bs


@torch.no_grad()
def apply_range_aware_correction(
    f: torch.Tensor,
    hat_delta: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Apply the fitted range-aware correction to produce final predictions.
    """

    mag = torch.tanh(alpha * hat_delta.abs())

    corrected_f = torch.where(
        hat_delta < 0,
        f - f * mag,
        f + (1.0 - f) * mag,
    )

    return corrected_f


def get_hat_Delta_bs_corr(
    f: torch.Tensor,
    y: torch.Tensor,
    hat_delta: torch.Tensor,
) -> torch.Tensor:
    bs = ((f - y) ** 2).cpu().numpy()
    abs_hat_delta = hat_delta.abs().cpu().numpy()
    corr = spearmanr(abs_hat_delta, bs)[0]
    corr = float(corr)
    return corr


@torch.no_grad()
def evaluate_kernel(phi, loader, loss_cfg, device, model_type="kernel"):
    phi.eval()
    agg = {}
    n = 0

    for batch in loader:
        emb = batch["emb"].to(device)
        f = batch["f"].to(device)
        y = batch["y"].to(device)
        x = batch["emb"].to(device)

        if model_type == "kernel_tokenwise":
            mask = batch["attention_mask"].to(device, non_blocking=True)
            emb = emb.squeeze(1)
            mask = mask.squeeze(1)
            z, attn_weights = phi(emb, mask=mask, return_attn=True)
            out = kernel_calib_discovery_loss(z=z, f=f, y=y, x=x, cfg=loss_cfg)
            attn_entropy = (
                -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
            )
            out["attn_entropy"] = attn_entropy
            out["loss"] += loss_cfg.lambda_attn_entropy * attn_entropy
        else:
            z = phi(emb)
            out = kernel_calib_discovery_loss(z=z, f=f, y=y, x=x, cfg=loss_cfg)

        bsz = emb.size(0)
        for k, v in out.items():
            agg[k] = agg.get(k, 0.0) + float(v.item()) * bsz
        n += bsz

    for k in agg:
        agg[k] /= max(n, 1)

    return agg


@torch.no_grad()
def evaluate_regression(phi, loader, cfg, device):
    phi.eval()
    agg = {}
    n = 0

    for batch in loader:
        emb = batch["emb"].to(device)
        f = batch["f"].to(device)
        y = batch["y"].to(device)
        if cfg.model.get("conditional", False):
            emb = torch.cat([emb, f.unsqueeze(1)], dim=1)
        pred_resid = phi(emb).flatten()
        out = regression_loss(
            pred_resid=pred_resid,
            f=f,
            y=y,
            cfg=cfg.loss,
            params=None,
        )
        bsz = emb.size(0)
        for k, v in out.items():
            agg[k] = agg.get(k, 0.0) + float(v.item()) * bsz
        n += bsz

    for k in agg:
        agg[k] /= max(n, 1)

    return agg


def get_extra_resreg_logs(val_pack):
    logs = {}
    pred_resid = val_pack["z"].flatten()
    if val_pack["has_eta"]:
        resid = val_pack["eta"] - val_pack["f"]
        corr = pearson_corr(pred_resid, resid)
        logs["corr_predRes_resid"] = float(corr.item())

    if val_pack["has_g"]:
        g = val_pack["g"]
        auc = auc_rank(pred_resid, g.float())
        auc_sym = torch.maximum(auc, 1.0 - auc) if torch.isfinite(auc) else auc
        logs["auc_predRes_predicts_g"] = (
            float(auc.item()) if torch.isfinite(auc) else float("nan")
        )
        logs["auc_sym_predRes_predicts_g"] = (
            float(auc_sym.item()) if torch.isfinite(auc_sym) else float("nan")
        )

        pred = (pred_resid > 0).long()
        acc = (pred == g).float().mean()
        acc_sym = torch.maximum(acc, 1.0 - acc)
        logs["sign_acc"] = float(acc.item())
        logs["sign_acc_sym"] = float(acc_sym.item())
    return logs


def get_extra_kernel_logs(val_pack, hat_delta):
    logs = {}
    if val_pack["has_eta"]:
        delta = (val_pack["eta"] - val_pack["f"]).float()
        pearson_c = pearson_corr(hat_delta, delta)
        spearman_c = spearmanr(hat_delta.cpu().numpy(), delta.cpu().numpy())[0]
        logs["pcorr_hatDelta_delta"] = float(pearson_c.item())
        logs["scorr_hatDelta_delta"] = float(spearman_c)
    if val_pack["has_g"]:
        g = val_pack["g"]
        auc = auc_rank(hat_delta, g.float())
        # If you only care about separability up to label flip:
        auc_sym = torch.maximum(auc, 1.0 - auc) if torch.isfinite(auc) else auc
        logs["auc_hatDelta_predicts_g"] = (
            float(auc.item()) if torch.isfinite(auc) else float("nan")
        )
        logs["auc_sym_hatDelta_predicts_g"] = (
            float(auc_sym.item()) if torch.isfinite(auc_sym) else float("nan")
        )

        # Also report sign-accuracy (again, only meaningful if g is aligned to over/under)
        pred = (hat_delta > 0).long()
        acc = (pred == g).float().mean()
        acc_sym = torch.maximum(acc, 1.0 - acc)
        logs["sign_acc"] = float(acc.item())
        logs["sign_acc_sym"] = float(acc_sym.item())
    return logs


def plot_binned_relplot(f, y, fig=None, ax=None, n_bins=10, cmap=None, norm=None):
    df = pd.DataFrame(
        {
            "f": f,
            "y": y,
        }
    )

    # 1. Build explicit equal-width bin edges
    edges = np.linspace(0, 1, n_bins + 1)

    # 2. Create an IntervalIndex
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")

    # 3. Bin using the IntervalIndex
    df["f_bin"] = pd.cut(df["f"], bins=intervals)

    # 2. Aggregate statistics per bin
    binned = (
        df.groupby("f_bin", observed=True)
        .agg(mean_y=("y", "mean"), count=("y", "size"))
        .reset_index()
    )

    # 3. Compute bin centers for equally spaced bars
    binned["bin_center"] = binned["f_bin"].apply(lambda x: x.mid)

    # 4. Normalize counts for coloring
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("cmap", ["lightgrey", "dimgrey"])
    if norm is None:
        norm = plt.Normalize(binned["count"].min(), binned["count"].max())
    colors = cmap(norm(binned["count"]))

    # 5. Use constant bar width (equal spacing)
    bin_width = 1.0 / n_bins

    # 6. Bin centers for visualization (the bar heights show empirical mean_y per bin)
    binned["bin_center"] = binned["bin_center"].astype(float)

    # Calibration metric: smoothECE (Blasiok & Nakkiran 2023). Bandwidth is
    # selected automatically by relplot via Silverman's rule. Computed on the
    # raw (f, y), independent of the histogram binning used for the bars.
    sece = get_smece(np.asarray(f), np.asarray(y))

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(2.8, 2.5))

    ax.bar(
        binned["bin_center"],
        binned["mean_y"],
        width=bin_width,
        color=colors,
        # edgecolor="black",
        align="center",
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="gray")

    # Colorbar for frequency
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # smECE annotation
    ax.text(
        0.05,
        0.9,
        r"$\mathrm{smECE}$ = %.3f" % sece,
        transform=ax.transAxes,
        bbox=dict(facecolor="white"),
    )

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("$f$")
    ax.set_ylabel(r"$\mathbb{E}[y \mid f]$")

    return fig, ax


def get_ece(f, y, n_bins=10):
    df = pd.DataFrame(
        {
            "f": f,
            "y": y,
        }
    )
    edges = np.linspace(0, 1, n_bins + 1)
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    df["f_bin"] = pd.cut(df["f"], bins=intervals)
    binned = (
        df.groupby("f_bin", observed=True)
        .agg(mean_y=("y", "mean"), count=("y", "size"))
        .reset_index()
    )
    binned["bin_center"] = binned["f_bin"].apply(lambda x: x.mid)
    N = len(df)
    binned["bin_center"] = binned["bin_center"].astype(float)
    ece = (binned["count"] / N * np.abs(binned["mean_y"] - binned["bin_center"])).sum()
    return ece


def get_smece(f, y):
    f = np.array(f)
    y = np.array(y)
    return float(relplot.smECE(f, y))
