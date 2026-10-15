import torch, torch.nn.functional as F


def regression_loss(pred_resid, f, y, cfg, params=None):
    """
    pred: model predictions of the residuals
    f: model confidences
    y: true labels
    """
    resid = y - f
    loss = F.mse_loss(pred_resid.float(), resid.float())
    # add l2 regularization
    if cfg.l2_reg > 0 and params is not None:
        l2_loss = 0.0
        for p in params:
            l2_loss += torch.sum(p**2)
        loss += cfg.l2_reg * l2_loss

    return {
        "loss": loss,
    }
