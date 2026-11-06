# OurNet

**Occlusion urban road extraction network: From global urban high-resolution benchmark dataset to guided generation method**

This repository contains the final **ConvNeXt_OccHeadv6** implementation for occlusion-aware road extraction. Baselines, ablations, datasets, experiment outputs and pretrained weights are excluded.

![OurNet architecture](docs/assets/architecture.png)

The model takes an RGB patch and an aligned precomputed diffusion feature. It predicts road/occlusion classes, complete roads and a boundary response. The diffusion feature extractor is **not included**: this release exposes its input interface, rather than providing an RGB-only end-to-end pipeline.

## Environment

Validated locally with Python 3.9. Python 3.9 is recommended for the pinned environment. Install a PyTorch/torchvision pair suitable for your CUDA runtime, then the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the versions used for local validation. CUDA is recommended for training. CPU inference is available with `--device cpu`. MMSegmentation, MMCV and MMEngine are not required.

## Data interface

Prepare a JSONL manifest with one sample per line. All paths are relative to the manifest (absolute paths also work):

```json
{"image":"images/example.png","label":"labels/example.png","feature":"features/example.pt"}
```

- **Image:** RGB image, resized to 512×512, normalized as `pixel / 127.5 - 1`.
- **Label:** single-channel or palette PNG with integer class IDs. Nearest-neighbor resizing preserves IDs. Three classes: `0` background, `1` visible road, `2` occluded road. Five classes: `0` background, `1` visible road, `2–4` occlusion subtypes; retain the source dataset's subtype mapping.
- **Feature:** PyTorch float tensor of shape `[1280,16,16]` (or `[1,1280,16,16]`) saved with `torch.save`. The feature must correspond to the same resized image, without independent geometric augmentation. The original cache uses `t_181_up-ft-index_0`; only index 0 is consumed by the final network. Other feature scales need not be loaded.

No images, labels or cached features are shipped. The feature extraction model, prompt and preprocessing must match those used to build the original cache; arbitrary tensors of the same size do not reproduce the paper.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --manifest data/train.jsonl --output outputs/ournet \
  --num-classes 3 --epochs 100 --batch-size 8 --seed 197 \
  --backbone-weights checkpoints/convnext_base_22k_224.pth
```

For five classes, use `--num-classes 5`. The optional backbone checkpoint is loaded from its `model` entry (or a plain state dict). Without it, training starts with random initialization. Paper-compatible initialization uses ConvNeXt-Base ImageNet-22K weights. Training uses Adam with learning rate `2e-4`, divided by five at 50%, 70% and 85% of the run. The final checkpoint is saved as `outputs/ournet/final.pth`; configuration and per-epoch losses are saved alongside it. This entry point uses one device; batch size is the total batch size on that device.

The loss combines multiclass cross-entropy/Dice, complete-road BCE/Dice, and L1 boundary supervision. **Important reproducibility detail:** the supplied final source data loader sets `dist_maps=0.0`, with distance-map generation commented out. This release preserves that zero target instead of silently inventing a boundary target. The loss's hardcoded five-class Dice setting is corrected to use the requested class count, and binary Dice has a small numerical safeguard. Thus this is a cleaned implementation, not a claim of exact retraining of published metrics. Actual boundary-distance supervision must be resolved against the original experiment configuration before claiming paper-equivalent retraining.

## Inference

```bash
python infer.py --image data/images/example.png \
  --feature data/features/example.pt \
  --checkpoint checkpoints/ConvNeXt_OccHeadv6_Occ2_100.th \
  --num-classes 3 --output outputs/example
```

The loader accepts original DataParallel checkpoints (`module.` prefixes), plain state dicts, and the new training checkpoint format. Class count must match the checkpoint. Full-model checkpoint loading is strict and does not download or require a separate backbone checkpoint.

Outputs (512×512 patch coordinates):

- `classes.png`: integer class IDs.
- `roads.png`: complete-road binary mask (0/255), combining non-background class predictions with the binary road head, as in the original evaluation.
- `boundary.npy`: floating-point boundary-head response.

Large georeferenced raster tiling/stitching and diffusion-feature generation are outside this release. Input georeferencing is not propagated to PNG outputs. Trained weights are not bundled in Git.

## Implementation provenance

The final model is extracted from `ConvNeXt_OccHeadv6`; its parameter names and tensor operations are retained for compatibility with the final experiment checkpoints. Local absolute paths, unused model imports and automatic backbone loading are removed. Input size remains 512×512, matching the original fixed-size heads.

The architecture figure was extracted from the author-provided manuscript. The ConvNeXt backbone is derived from Meta's ConvNeXt; its MIT notice is preserved in [LICENSE-ConvNeXt](LICENSE-ConvNeXt). See [THIRD_PARTY.md](THIRD_PARTY.md). This repository does not grant rights to external model weights or datasets.
