"""mIoU через матрицу ошибок (классовое усреднение, void/ignore пропускается)."""
import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.n = num_classes
        self.ignore = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = pred.flatten().cpu(), target.flatten().cpu()
        keep = target != self.ignore
        idx = target[keep] * self.n + pred[keep]
        self.mat += torch.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def compute(self):
        m = self.mat.double()
        tp = m.diag()
        union = m.sum(0) + m.sum(1) - tp
        iou = tp / union.clamp(min=1)
        present = m.sum(1) > 0  # класс есть в GT; отсутствующие не входят в среднее
        iou_np = iou.numpy().copy()
        iou_np[~present.numpy()] = np.nan
        return {
            "mIoU": float(np.nanmean(iou_np)),
            "pixel_acc": float(tp.sum() / m.sum().clamp(min=1)),
            "iou_per_class": iou_np,
        }
