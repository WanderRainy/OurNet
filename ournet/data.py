"""Manifest-based interface; no dataset files are distributed."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


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
    def __init__(self, manifest, num_classes):
        self.path = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        self.num_classes = num_classes
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
        # Matches the supplied final training loader; see README limitations.
        return inputs, {'label': label, 'dist_maps': torch.zeros(1, 512, 512)}
