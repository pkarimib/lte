import yaml
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


class SuperResolutionModel():
    def __init__(self, config_path, checkpoint='None'):
        super(SuperResolutionModel, self).__init__()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # config parameters
        generator_params = config['model_params']['generator_params']
        self.shape = config['dataset_params']['frame_shape']
        self.use_lr_video = generator_params.get('use_lr_video', True)
        self.lr_size = generator_params.get('lr_size', 256)
        self.generator_type = generator_params.get('generator_type', 'swinir-lte')
        self.scale = int(self.shape[1] / self.lr_size)
        scale_max = 4

        h = int(self.lr_size * int(self.scale))
        w = int(self.lr_size * int(self.scale))

        self.coord = make_coord((h, w))
        if torch.cuda.is_available:
            self.coord = self.coord.cuda()

        self.cell = torch.ones_like(self.coord)
        self.cell[:, 0] *= 2 / h
        self.cell[:, 1] *= 2 / w
        
        self.cell_factor = max(self.scale/scale_max, 1)

        # initialize weights
        if checkpoint == 'None':
            checkpoint = config['checkpoint_params']['checkpoint_path']
        print(checkpoint)

        self.generator = models.make(torch.load(checkpoint)['model'], load_sd=True)
        if torch.cuda.is_available():
            self.generator = self.generator.cuda()

        timing_enabled = True
        self.times = []

    def get_shape(self):
        return tuple(self.shape)


    def get_lr_video_info(self):
        return self.use_lr_video, self.lr_size


    def predict_with_lr_video(self, img):
        """ predict and return the target RGB frame
            from a low-res version of it.
        """
        pred = batched_predict(self.generator, 
                ((img - 0.5) / 0.5).cuda().unsqueeze(0),
                self.coord.unsqueeze(0), self.cell_factor*self.cell.unsqueeze(0),
                bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(self.shape[0], self.shape[1], 3).permute(2, 0, 1).cpu()
        transforms.ToPILImage()(pred).save("wrapper.png")
        return True
