"""Manifest-based interface; no dataset files are distributed."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from scipy.ndimage import binary_erosion, distance_transform_edt


def boundary_distance_target(label, sigma=3.0):
    """Gaussian of Euclidean distance to the complete-road inner boundary.

    Labels > 0 form one road mask; occlusion/visible interfaces are not edges.
    Image-frame borders alone do not create a boundary. Empty/full masks
    with no observable road/nonroad interface receive a zero target.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Boundary sigma must be finite and positive")
    road = np.asarray(label) > 0
    if road.ndim != 2:
        raise ValueError("Expected a two-dimensional label map")
    boundary = road & ~binary_erosion(road, border_value=1)
    if not boundary.any():
        return np.zeros(road.shape, dtype=np.float32)
    distance = distance_transform_edt(~boundary)
    return np.exp(-(distance ** 2) / (2 * sigma ** 2)).astype(np.float32)


def load_image(path):
    image = Image.open(path).convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 127.5 - 1


def load_feature(path):
    feature = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(feature, torch.Tensor):
        raise ValueError(f'{path}: expected a tensor')
    if feature.shape == (1, 1280, 16, 16):
        feature = feature[0]
    if feature.shape != (1280, 16, 16):
        raise ValueError(f'{path}: expected [1280,16,16], got {tuple(feature.shape)}')
    if not torch.isfinite(feature).all():
        raise ValueError(f'{path}: feature contains nonfinite values')
    return feature.float()


class RoadDataset(Dataset):
    """JSONL fields: image, feature, label; paths relative to manifest."""
    def __init__(self, manifest, num_classes, boundary_sigma=3.0):
        self.path = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        self.num_classes = num_classes
        if not np.isfinite(boundary_sigma) or boundary_sigma <= 0:
            raise ValueError("Boundary sigma must be finite and positive")
        self.boundary_sigma = boundary_sigma
        if not self.rows:
            raise ValueError('Empty manifest')
        for row in self.rows:
            for key in ('image', 'feature', 'label'):
                if key not in row or not self.resolve(row[key]).is_file():
                    raise ValueError(f'Missing {key}: {row}')

    def resolve(self, path):
        return self.path.parent / path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        mask_image = Image.open(self.resolve(row['label']))
        if mask_image.mode not in ('L', 'P', 'I', 'I;16'):
            raise ValueError('Labels must contain integer class IDs, not RGB colors')
        mask = np.array(mask_image.resize((512, 512), Image.Resampling.NEAREST), dtype=np.int64)
        if mask.min() < 0 or mask.max() >= self.num_classes:
            raise ValueError(f'Label IDs must be in [0, {self.num_classes-1}]')
        label = torch.from_numpy(mask.copy()).unsqueeze(0)
        inputs = {'img': load_image(self.resolve(row['image'])),
                  'img_sdpt': [load_feature(self.resolve(row['feature']))]}
        distance_target = torch.from_numpy(boundary_distance_target(mask, self.boundary_sigma)).unsqueeze(0)
        return inputs, {'label': label, 'dist_maps': distance_target}
