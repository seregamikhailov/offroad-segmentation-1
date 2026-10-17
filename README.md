# offroad-segmentation

CNN-based semantic segmentation of off-road scenes on the [RUGD](http://rugd.vision/) dataset: a software component for data preparation, model training, quality evaluation and model optimization.

> **Status:** work in progress (bachelor thesis project). The baseline training pipeline is in place; experiments and results will be added as they are completed.

## Scope

- Data preparation for RUGD (annotation conversion, sequence-based train/val/test split, class-imbalance analysis).
- Training and evaluation of open-source convolutional segmentation models (FCN, U-Net, DeepLabV3+ and others available in `segmentation_models_pytorch`).
- Optimization of the training setup: encoder choice and model complexity, loss functions and class weights for rare classes, optimizer and learning-rate schedule, augmentations.
- Not in scope: inference serving and deployment.

The project deliberately relies only on openly distributed models and libraries (no YOLO/SAM-family models).

## Repository layout

```
configs/
  baseline.yaml        baseline experiment (DeepLabV3+ / ResNet34)
  rugd_split.json      sequence-to-split assignment used for train/val/test
scripts/
  download_rugd_hf.py  download RUGD from a Hugging Face mirror
  prepare_rugd.py      colour masks -> class-id masks, split lists, classes.json
  search_split.py      search for a class-balanced split by video sequence
src/offroad_seg/
  data.py              dataset and augmentations (Albumentations)
  models.py            model factory, device selection
  losses.py            cross-entropy, CE + Dice, inverse-frequency class weights
  metrics.py           confusion-matrix based mIoU / pixel accuracy (void ignored)
  train.py             training loop, per-epoch log, best checkpoint by val mIoU
```

## Setup

Developed on macOS (Apple Silicon, MPS backend) with Python 3.14; CUDA and CPU are selected automatically when available.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data

RUGD (Wigness et al., IROS 2019) is **not** included in this repository. Obtain it from the [official site](http://rugd.vision/) (frames and annotations archives). The official host was unreliable when this project started, so `scripts/download_rugd_hf.py` fetches an unofficial community mirror (`WilliamBonilla62/RUGD` on Hugging Face) that keeps the original folder layout. Check the dataset's terms of use before reusing it; the mirror carries no license of its own.

```bash
python scripts/download_rugd_hf.py     # ~5.7 GB into data/raw/extracted
python scripts/prepare_rugd.py         # writes data/rugd/{masks,splits,classes.json}
```

### Split

The official RUGD split file was not available, so frames are split **by video sequence** (neighbouring frames of one video are nearly identical, so a per-frame random split leaks information). `scripts/search_split.py` searches for an assignment that keeps every class present in all splits where possible; the result is stored in `configs/rugd_split.json` (train 3570 / val 1923 / test 1943 frames).

Known limitations: `bicycle`, `person` and `bridge` are practically absent from the training split (`bridge` appears only in test), so their IoU is expected to be about zero. Results are therefore **not directly comparable** with papers that use the official split. Pixels labelled `void` are ignored in the loss and in all metrics.

## Training

```bash
# quick sanity run
python -m offroad_seg.train --config configs/baseline.yaml --limit 64 --epochs 1

# full baseline run
python -m offroad_seg.train --config configs/baseline.yaml
```

The first run with `encoder_weights: imagenet` downloads the ImageNet-pretrained encoder weights automatically. Outputs go to `runs/<name>/`: `config.yaml`, `log.csv` (loss, mIoU, pixel accuracy per epoch), `best.pt` (best checkpoint by validation mIoU) and `best_iou_per_class.json`.

Evaluation protocol: the model is selected on the validation split; final numbers are reported on the test split only.

## Roadmap

1. Baseline: DeepLabV3+ / ResNet34, mIoU on val and test, per-class IoU.
2. Loss and class-imbalance experiments (class weights, Focal, Lovász).
3. Encoder comparison by model complexity (ResNet18/34, MobileNetV2, EfficientNet-B0).
4. Hyperparameter and augmentation tuning.
5. Final evaluation of the best configurations on the test split.

## License

MIT, see [LICENSE](LICENSE).

## References

- M. Wigness, S. Eum, J. G. Rogers, D. Han, H. Kwon. *A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments.* IROS 2019. DOI: 10.1109/IROS40897.2019.8968283
- L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam. *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.* ECCV 2018. arXiv:1802.02611
