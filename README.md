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
  sweeps/screening.yaml  25-run screening sweep (architecture, encoder, loss, recipe)
scripts/
  download_rugd_hf.py  download RUGD from a Hugging Face mirror
  prepare_rugd.py      colour masks -> class-id masks, split lists, classes.json
  search_split.py      search for a class-balanced split by video sequence
  check_sweep.py       pre-flight check: build every model of a sweep, one forward/backward pass
  setup_remote.sh      one-shot setup on a Linux + NVIDIA GPU machine
  run_screening.sh     run the screening sweep and aggregate results
src/offroad_seg/
  data.py              dataset and augmentations (Albumentations)
  models.py            model factory, device selection
  losses.py            CE, focal, Dice, Lovasz and their combinations, inverse-frequency class weights
  metrics.py           confusion-matrix based mIoU / pixel accuracy (void ignored)
  train.py             training loop, optional early stopping, best checkpoint by val mIoU, summary.json
  evaluate.py          evaluate a checkpoint on a split (test only for final models)
  sweep.py             run the experiments of a YAML sweep file (restartable)
  aggregate.py         collect all summary.json files into results.csv / per_class_iou.csv
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

## Baseline result

DeepLabV3+ / ResNet34, ImageNet encoder, CE + Dice, 384x512, 30 epochs, one run (seed 42):

| Split | mIoU | Pixel accuracy |
|---|---|---|
| val (best epoch 21) | 36.5% | 88.1% |
| test | 30.1% | 75.0% |

Frequent classes (tree, grass, sky, gravel, mulch) are segmented well; rare and small classes (dirt, container, rock-bed, bicycle, person, sign) are close to zero. Per-class IoU differs strongly between val and test (for example asphalt 0.91 vs 0.07), because rare classes occupy a tiny and different part of each split. Differences of a few mIoU points between single runs should therefore be treated with care.

## Experiments

The screening sweep (`configs/sweeps/screening.yaml`) changes one factor at a time against the baseline, with the same 20-epoch schedule and one seed per run (25 runs):

| Group | Variants |
|---|---|
| `arch` | DeepLabV3+, DeepLabV3, U-Net, U-Net++, FPN, PSPNet, Linknet, MAnet, PAN, UPerNet (ResNet34 encoder) |
| `encoder` | ResNet18/50/101, MobileNetV2, EfficientNet-B0 (DeepLabV3+) |
| `loss` | CE, class-weighted CE+Dice, Focal+Dice, Lovasz, CE+Lovasz |
| `recipe` | training from scratch, 288x384 and 512x672 input, learning rate 1e-4 and 1e-3 |

```bash
python scripts/check_sweep.py --sweep configs/sweeps/screening.yaml --device cuda   # pre-flight
python -m offroad_seg.sweep --sweep configs/sweeps/screening.yaml                   # all runs, restartable
python -m offroad_seg.sweep --sweep configs/sweeps/screening.yaml --group loss      # one group
python -m offroad_seg.aggregate --runs runs/screening --out results/screening       # results table
python -m offroad_seg.evaluate --weights runs/screening/<run>/best.pt --split test  # final models only
```

Models are compared on val; the test split is evaluated only for the few selected final configurations.

### Running on a remote GPU

```bash
git clone git@github.com:seregamikhailov/offroad-segmentation-1.git && cd offroad-segmentation-1
bash scripts/setup_remote.sh          # venv, dependencies, RUGD download, split, pre-flight check
nohup bash scripts/run_screening.sh > sweep.log 2>&1 &
```

If the machine has many CPU cores, raise `data.num_workers` under `common:` in the sweep file (default is 4).

## Roadmap

1. Screening sweep (above): which architecture, encoder, loss and training recipe matter most.
2. Repeat the best configurations with several seeds to measure run-to-run variance.
3. Targeted work on rare classes (weights, sampling, augmentation) and on model complexity vs. quality.
4. Final evaluation of the selected configurations on the test split.

## License

MIT, see [LICENSE](LICENSE).

## References

- M. Wigness, S. Eum, J. G. Rogers, D. Han, H. Kwon. *A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments.* IROS 2019. DOI: 10.1109/IROS40897.2019.8968283
- L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam. *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.* ECCV 2018. arXiv:1802.02611
