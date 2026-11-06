"""Run patch inference with RGB imagery and an aligned diffusion feature."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ournet import OurNet
from ournet.data import load_image, load_feature
from ournet.checkpoint import load_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--feature', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--num-classes', type=int, choices=(3, 5), default=3)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    model = OurNet(a.num_classes)
    load_weights(model, a.checkpoint)
    model.to(a.device).eval()
    inputs = {'img': load_image(a.image).unsqueeze(0).to(a.device),
              'img_sdpt': [load_feature(a.feature).unsqueeze(0).to(a.device)]}
    with torch.inference_mode(): result = model(inputs)
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    classes = result['roadocc_masks'].argmax(1)[0].cpu().numpy().astype(np.uint8)
    road_head = (result['roadwo_masks'][0, 0] >= .5).cpu().numpy()
    road = ((classes > 0) | road_head).astype(np.uint8)*255
    boundary = result['roadwo_edge_masks'][0, 0].cpu().numpy()
    Image.fromarray(classes).save(output/'classes.png')
    Image.fromarray(road).save(output/'roads.png')
    np.save(output/'boundary.npy', boundary)


if __name__ == '__main__': main()
