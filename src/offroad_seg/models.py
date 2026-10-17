"""Фабрика моделей на базе segmentation_models_pytorch."""
import segmentation_models_pytorch as smp
import torch


def build_model(arch: str, encoder: str, num_classes: int, encoder_weights=None):
    return smp.create_model(
        arch, encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=num_classes
    )


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
