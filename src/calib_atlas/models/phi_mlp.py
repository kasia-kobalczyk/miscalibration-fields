import torch, torch.nn as nn, torch.nn.functional as F


class PhiMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.normalize = getattr(cfg, "normalize", False)
        hidden_dim = cfg.hidden_dim
        num_hidden = max(int(getattr(cfg, "num_hidden_layers", 2)), 0)

        layers = []
        if cfg.get("conditional", False):
            in_dim = cfg.in_dim + 1
        else:
            in_dim = cfg.in_dim
        if num_hidden == 0:
            layers.append(nn.Linear(in_dim, cfg.out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_hidden - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, cfg.out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, emb):
        z = self.net(emb)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z


class TokenAttentionPooling(nn.Module):
    """
    Learns attention weights over token embeddings.

    Input:
        emb: [bs, T, dim_model]
        mask: [bs, T] (optional, 1 for valid tokens, 0 for padding)

    Output:
        pooled: [bs, dim_model]
        attn_weights: [bs, T]
    """

    def __init__(self, dim_model, attn_hidden_dim=128, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.score_net = nn.Linear(dim_model, 1, bias=False)

    def forward(self, emb, mask=None):
        # emb: [bs, T, dim_model]

        scores = self.score_net(emb).squeeze(-1)  # [bs, T]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores / self.temperature, dim=-1)  # [bs, T]

        pooled = torch.sum(attn_weights.unsqueeze(-1) * emb, dim=1)  # [bs, dim_model]

        return pooled, attn_weights


class TokenwisePhi(nn.Module):
    """
    Full model:
        token embeddings -> attention pooling -> PhiMLP
    """

    def __init__(self, cfg):
        super().__init__()

        self.pool = TokenAttentionPooling(
            dim_model=cfg.in_dim,
            attn_hidden_dim=getattr(cfg, "attn_hidden_dim", 128),
            temperature=getattr(cfg, "attn_temperature", 1.0),
        )

        self.phi = PhiMLP(cfg)

    def forward(self, emb, mask=None, return_attn=False):
        """
        emb: [bs, T, dim_model]
        mask: [bs, T] optional
        """
        pooled, attn_weights = self.pool(emb, mask)
        z = self.phi(pooled)

        if return_attn:
            return z, attn_weights
        return z
