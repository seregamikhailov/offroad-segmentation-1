import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class CEDiceLoss(nn.Module):
    """Cross-entropy (с игнорированием void) + Dice; веса классов — опционально."""

    def __init__(self, ignore_index=255, class_weights=None, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss("multiclass", ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(kind: str, ignore_index=255, class_weights=None):
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    if kind == "ce_dice":
        return CEDiceLoss(ignore_index, class_weights)
    raise ValueError(f"unknown loss: {kind}")


def inverse_frequency_weights(freq, eps=1e-3, clip=(0.2, 10.0)):
    """w_c = 1/sqrt(freq_c), нормированные на среднее и обрезанные."""
    w = 1.0 / torch.sqrt(torch.tensor(freq, dtype=torch.float32) + eps)
    w = w / w.mean()
    return w.clamp(*clip)
