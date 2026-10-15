import torch, torch.nn.functional as F


def kernel_calib_discovery_loss(z, f, y, x, cfg):
    B = z.size(0)
    # Delta Loss
    r = (y - f).view(B, 1).float()
    d2 = torch.cdist(z, z) ** 2
    K = torch.exp(-d2 / (cfg.sigma**2 + cfg.eps))
    if cfg.tau is not None:
        df2 = (f[:, None] - f[None, :]) ** 2
        K = K * torch.exp(-df2 / (cfg.tau**2 + cfg.eps))
    K = K * (1 - torch.eye(B, device=z.device))
    m = K.sum(1, keepdim=True)
    Delta = (K @ r) / (m + cfg.eps)
    loss_detect = -(Delta.squeeze() ** 2).mean()

    loss = loss_detect

    # Penalty for effective sample size
    if cfg.lambda_mass > 0:
        loss_mass = (F.relu(cfg.min_mass - m.squeeze()) ** 2).mean()
        loss += cfg.lambda_mass * loss_mass
    else:
        loss_mass = torch.tensor(0.0, device=z.device)

    # Variance Penalty
    if cfg.lambda_var > 0:
        var = z.var(0, unbiased=False)
        loss_var = F.relu(cfg.var_floor - var).mean()
        loss += cfg.lambda_var * loss_var
    else:
        loss_var = torch.tensor(0.0, device=z.device)

    # Semantic regularization
    if cfg.lambda_semantic > 0:
        # Encourage nearby points in x-space to also be nearby in z-space
        d2_x = torch.cdist(x, x) ** 2
        K_x = torch.exp(-d2_x / (cfg.sigma_semantic**2 + cfg.eps))
        if cfg.weighted_semantic:
            # focus on points close in the x space
            weights = K_x
            loss_semantic = weights * (K - K_x).pow(2)
            # zero out the diagonal terms where i=j
            loss_semantic = loss_semantic * (1 - torch.eye(B, device=z.device))
            weights = weights * (1 - torch.eye(B, device=weights.device))
            loss_semantic = loss_semantic.sum() / weights.sum()
        else:
            loss_semantic = F.mse_loss(K, K_x)

        loss += cfg.lambda_semantic * loss_semantic
    else:
        loss_semantic = torch.tensor(0.0, device=z.device)

    return {
        "loss": loss,
        "loss_detect": loss_detect,
        "loss_mass": loss_mass,
        "loss_var": loss_var,
        "loss_semantic": loss_semantic,
        "mean_mass": m.mean(),
        "mean_abs_Delta": Delta.abs().mean(),
        "mean_Delta2": (Delta.squeeze() ** 2).mean(),
    }
