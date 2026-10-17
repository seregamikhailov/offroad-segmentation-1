import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Multiclass focal loss, FL = -(1 - p_t)^gamma * log(p_t), averaged over non-void pixels.

    Own vectorised implementation (smp's version loops over classes and fails on Apple MPS).
    """

    def __init__(self, gamma=2.0, ignore_index=255, weight=None):
        super().__init__()
        self.gamma, self.ignore_index = gamma, ignore_index
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        valid = target != self.ignore_index
        safe = target.masked_fill(~valid, 0)
        logp = torch.log_softmax(logits.float(), dim=1).gather(1, safe.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - logp.exp()) ** self.gamma) * logp
        if self.weight is not None:
            loss = loss * self.weight[safe]
        return loss[valid].sum() / valid.sum().clamp(min=1)


class CompoundLoss(nn.Module):
    """Weighted sum of a pixel-wise term and a region-based term (both ignore void)."""

    def __init__(self, first: nn.Module, second: nn.Module, second_weight: float = 1.0):
        super().__init__()
        self.first, self.second, self.second_weight = first, second, second_weight

    def forward(self, logits, target):
        return self.first(logits, target) + self.second_weight * self.second(logits, target)


def build_loss(kind: str, ignore_index=255, class_weights=None, focal_gamma=2.0):
    """kind: ce | ce_dice | focal | focal_dice | lovasz | ce_lovasz"""
    ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=ignore_index)
    focal = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index, weight=class_weights)
    lovasz = smp.losses.LovaszLoss("multiclass", ignore_index=ignore_index)
    table = {
        "ce": lambda: ce,
        "focal": lambda: focal,
        "lovasz": lambda: lovasz,
        "ce_dice": lambda: CompoundLoss(ce, dice),
        "focal_dice": lambda: CompoundLoss(focal, dice),
        "ce_lovasz": lambda: CompoundLoss(ce, lovasz),
    }
    if kind not in table:
        raise ValueError(f"unknown loss '{kind}', expected one of {sorted(table)}")
    return table[kind]()


def inverse_frequency_weights(freq, eps=1e-3, clip=(0.2, 10.0)):
    """w_c = 1/sqrt(freq_c), normalised to mean 1 and clipped."""
    w = 1.0 / torch.sqrt(torch.tensor(freq, dtype=torch.float32) + eps)
    w = w / w.mean()
    return w.clamp(*clip)
