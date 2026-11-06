"""Train the final ConvNeXt OurNet variant from a JSONL manifest."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ournet import ConvNeXt_OccHeadv6
from ournet.data import RoadDataset
from ournet.loss import B2B1E1Loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--backbone-weights', help='Optional ConvNeXt-Base pretrained checkpoint')
    p.add_argument('--num-classes', type=int, choices=(3, 5), default=3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=197)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1: p.error('epochs and batch-size must be positive')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    if (output/'final.pth').exists(): raise FileExistsError('Use a new output directory')
    dataset = RoadDataset(a.manifest, a.num_classes)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=a.workers)
    model = ConvNeXt_OccHeadv6(a.num_classes, a.backbone_weights).to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = B2B1E1Loss(num_classes=a.num_classes)
    (output/'config.json').write_text(json.dumps(vars(a), indent=2))
    for epoch in range(1, a.epochs+1):
        model.train(); total = 0.
        for inputs, targets in loader:
            inputs = {'img': inputs['img'].to(a.device), 'img_sdpt': [v.to(a.device) for v in inputs['img_sdpt']]}
            targets = {k: v.to(a.device) for k, v in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimizer.step(); total += loss.item()
        record = {'epoch': epoch, 'loss': total/len(loader), 'lr': optimizer.param_groups[0]['lr']}
        print(json.dumps(record), flush=True)
        with (output/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if epoch in {int(.5*a.epochs), int(.7*a.epochs), int(.85*a.epochs)}:
            for group in optimizer.param_groups: group['lr'] /= 5
    torch.save({'model': model.state_dict(), 'num_classes': a.num_classes, 'epoch': a.epochs}, output/'final.pth')


if __name__ == '__main__': main()
