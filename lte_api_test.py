import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict
from lte_wrapper import SuperResolutionModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--checkpoint')
    parser.add_argument('--config')
    parser.add_argument('--output', default='output.png')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))
    model = SuperResolutionModel(args.config, args.checkpoint)
    model.predict_with_lr_video(img)
