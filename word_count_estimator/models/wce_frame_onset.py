from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def _lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    B = lengths.size(0)
    T = int(max_len or lengths.max().item())
    ar = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
    return ar < lengths.unsqueeze(1)

def _make_norm_1d(num_ch: int, spec: str) -> nn.Module:
    spec = (spec or "ln").lower()
    if spec == "bn":
        return MaskedBatchNorm1d(num_ch)
    if spec == "ln":
        return nn.GroupNorm(1, num_ch)
    if spec.startswith("gn:"):
        g = int(spec.split(":")[1])
        return nn.GroupNorm(g, num_ch)
    return nn.Identity()

class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias',   None)
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var',  torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var',  None)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if (not self.training) or (mask is None):
            if self.track_running_stats:
                mean = self.running_mean.view(1, -1, 1)
                var  = self.running_var.view(1, -1, 1)
                x_hat = (x - mean) / (var + self.eps).sqrt()
            else:
                mean = x.mean(dim=(0,2), keepdim=True)
                var  = x.var(dim=(0,2), unbiased=False, keepdim=True)
                x_hat = (x - mean) / (var + self.eps).sqrt()
        else:
            m = mask.to(x.dtype).unsqueeze(1)
            denom = m.sum(dim=(0,2))
            denom = torch.clamp(denom, min=1.0)
            sum_x  = (x * m).sum(dim=(0,2))
            mean   = sum_x / denom
            sum_sq = ((x - mean.view(1,-1,1))**2 * m).sum(dim=(0,2))
            var    = sum_sq / denom
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
                    self.num_batches_tracked += 1
            x_hat = (x - mean.view(1,-1,1)) / (var.view(1,-1,1) + self.eps).sqrt()
        if self.affine:
            x_hat = x_hat * self.weight.view(1,-1,1) + self.bias.view(1,-1,1)
        return x_hat


class _DWSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout, norm_spec="ln"):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.norm = _make_norm_1d(out_ch, norm_spec)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        if isinstance(self.norm, MaskedBatchNorm1d):
            x = self.norm(x, mask)
        else:
            x = self.norm(x)
        x = F.gelu(x)
        return self.drop(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C, T]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
        
class _TCNBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float, norm: str):
        super().__init__()
        self.c1 = _DWSeparableConv1d(ch, ch, kernel, dilation, dropout, norm)
        self.c2 = _DWSeparableConv1d(ch, ch, kernel, dilation, dropout, norm)

        self.se = SELayer(ch, reduction=8)

        self.post = _make_norm_1d(ch, norm)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        r = x
        x = self.c1(x, mask)
        x = self.c2(x, mask)
        x = x + r

        x = self.se(x)

        return x


class _Encoder(nn.Module):
    def __init__(self, in_feats: int, d_model: int, n_blocks: int, kernel: int, dropout: float,
                 use_bilstm: bool, lstm_hidden: int, lstm_layers: int, norm: str):
        super().__init__()
        self.in_proj = nn.Linear(in_feats, d_model)
        self.tcn = nn.ModuleList([
            _TCNBlock(d_model, kernel, 2 ** i, dropout, norm) for i in range(n_blocks)
        ])
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.rnn = nn.LSTM(
                input_size=d_model,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=0.0,
                bidirectional=True,
            )
            self.rnn_out = nn.Linear(2 * lstm_hidden, d_model)
        else:
            self.rnn = None
            self.rnn_out = nn.Identity()
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.in_proj(x)
        h = x.transpose(1, 2)
        mask = None
        if lengths is not None:
            mask = _lengths_to_mask(lengths, h.size(-1))
        for blk in self.tcn:
            h = blk(h, mask)
        x = h.transpose(1, 2)
        if self.use_bilstm:
            if lengths is not None:
                T = x.size(1)
                lens = lengths.detach().to('cpu', dtype=torch.long).clamp(min=1, max=T)
                packed = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
                packed_out, _ = self.rnn(packed)
                x, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            else:
                x, _ = self.rnn(x)
            x = self.rnn_out(x)
        return self.out_norm(x)


class WCEFrameOnset(nn.Module):
    def __init__(self,
                 in_feats: int = 80,
                 d_model: int = 128,
                 tcn_blocks: int = 8,
                 tcn_kernel: int = 5,
                 tcn_dropout: float = 0.3,
                 norm: str = "ln",
                 use_bilstm: bool = True,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1,
                 **kwargs):
        super().__init__()
        self.encoder = _Encoder(in_feats, d_model, tcn_blocks, tcn_kernel, tcn_dropout,
                                use_bilstm, lstm_hidden, lstm_layers, norm)
        self.frame_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, 1, 1)
        )

    def forward(self, feats: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, _ = feats.shape  # noqa: F841
        h = self.encoder(feats, lengths)
        logits = self.frame_head(h.transpose(1, 2)).squeeze(1)
        mask = None
        if lengths is not None:
            mask = _lengths_to_mask(lengths, T)
            logits = logits.masked_fill(~mask, -1e4)
        probs = torch.sigmoid(logits)
        if mask is not None:
            probs = probs * mask.float()
        count = probs.sum(dim=-1)
        return {"frame_logits": logits, "frame_probs": probs, "count": count}



# ------------------------------------------------------------------
# Losses functions
# ------------------------------------------------------------------

class CountL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, pred_count: torch.Tensor, true_count: torch.Tensor) -> torch.Tensor:
        return self.loss(pred_count, true_count.float())

class CountL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred_count: torch.Tensor, true_count: torch.Tensor) -> torch.Tensor:
        return self.loss(pred_count, true_count.float())

# ------------------------------------------------------------------
# Frame onset BCE loss with optional pos_weight and length masking
# pred_logits: (B, T)
# targets: (B, T) in {0,1}
# lengths: (B,) valid frame counts
# pos_weight: scalar tensor or float for rare positive balancing
# reduction: 'mean' or 'sum'
# ------------------------------------------------------------------
def frame_onset_bce_loss(pred_logits: torch.Tensor,
                         targets: torch.Tensor,
                         lengths: torch.Tensor,
                         pos_weight: Optional[torch.Tensor | float] = None,
                         reduction: str = 'mean') -> torch.Tensor:
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight), device=pred_logits.device)
    B, T = pred_logits.shape
    if targets.shape != pred_logits.shape:
        raise ValueError(f"targets shape {targets.shape} must match pred_logits {pred_logits.shape}")

    if lengths is None:
        lengths = torch.full((B,), T, device=pred_logits.device, dtype=torch.long)

    # build mask
    rng = torch.arange(T, device=pred_logits.device).unsqueeze(0)
    mask = (rng < lengths.unsqueeze(1)).float()
    # BCE with logits per frame

    if pos_weight is not None:
        # manual BCE to apply pos_weight: loss = -[ y*logsig * pos_weight + (1-y)*log(1-sig) ]
        # Use stable formulation with logits
        # BCEWithLogits: max(x,0) - x*y + log(1+exp(-abs(x))) ; incorporate pos_weight on positive term
        x = pred_logits
        y = targets
        max_val = torch.clamp(-x, min=0)
        logexp = torch.log1p(torch.exp(-torch.abs(x)))
        # Positive term scaled
        loss = (1 - y) * (x + max_val + logexp) + y * (pos_weight * (max_val + logexp) - pos_weight * x)

    else:
        loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')

    loss = loss * mask
    if reduction == 'sum':
        return loss.sum()

    # mean over valid frames
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


# ------------------------------------------------------------------
# Mixed frame BCE + count regression (L1/L2) loss
# ------------------------------------------------------------------
def frame_onset_bce_count_mix(
    out_dict: dict,
    frame_targets: torch.Tensor,
    lengths: torch.Tensor,
    pos_weight: Optional[float] = None,
    frame_weight: float = 1.0,
    count_weight: float = 0.01,
    count_loss_type: str = "l2",
) -> torch.Tensor:
    """Combine frame-level BCE (with optional pos_weight) and count regression loss.

    Args:
        out_dict: model forward output containing 'frame_logits' and 'count'.
        frame_targets: (B,T) binary frame onset labels.
        lengths: (B,) valid frame lengths.
        pos_weight: positive weight for BCE.
        frame_weight: weight for frame BCE term.
        count_weight: weight for count regression term.
        count_loss_type: 'l1' | 'l2'.
    Returns:
        Scalar mixed loss.
    """
    logits = out_dict["frame_logits"]

    #bce = frame_onset_bce_loss(logits, frame_targets, lengths, pos_weight=pos_weight, reduction='mean')
    bce = frame_onset_bce_loss(logits, frame_targets, lengths, pos_weight=pos_weight, reduction='sum')

    with torch.no_grad():
        true_count = frame_targets.sum(dim=1).float()
    pred_count = out_dict.get("count", torch.sigmoid(logits).sum(dim=1))

    if count_loss_type == "l1":
        cnt_loss = F.l1_loss(pred_count, true_count, reduction='mean')
    elif count_loss_type == "l2":
        cnt_loss = F.mse_loss(pred_count, true_count, reduction='mean')
    else:
        raise ValueError(f"Unsupported count_loss_type: {count_loss_type}")

    return frame_weight * bce + count_weight * cnt_loss


__all__ = ["WCEFrameOnset", 
           "CountL1Loss", "CountL2Loss", 
           "frame_onset_bce_loss", "frame_onset_bce_count_mix"]
