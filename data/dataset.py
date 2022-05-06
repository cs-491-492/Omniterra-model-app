import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler
from ever.api.data import distributed, CrossValSamplerGenerator
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    Background=(211,211,211),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 165, 0),
)

LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)

INT_TO_LABEL = {0:'Background', 1:'Building', 2:'Road', 3:'Water', 4:'Barren', 5:'Forest', 6:'Agricultural'}
LABEL_TO_HEX = {'Background': '#d3d3d3', 'Building': '#ff0000', 'Road': '#ffff00', 'Water': '#0000ff', 'Barren': '#a52a2a', 'Forest': '#00ff00', 'Agricultural': '#ffa500'}
HEX_TO_HSL = {'#d3d3d3': 'hsl(0,0%,83%)', '#ff0000': 'hsl(0,100%,50%)', '#ffff00': 'hsl(60,100%,50%)', '#0000ff': 'hsl(240,100%,50%)', '#a52a2a': 'hsl(0,0%,40%)', '#00ff00': 'hsl(120,100%,50%)', '#ffa500': 'hsl(30,100%,50%)'}
LABEL_TO_HSL = {'Background': 'hsl(0,0%,83%)', 'Building': 'hsl(120,0%,50%)', 'Road': 'hsl(60,100%,50%)', 'Water': 'hsl(240,100%,50%)', 'Barren': 'hsl(0,0%,40%)', 'Forest': 'hsl(120,100%,50%)', 'Agricultural': 'hsl(30,100%,50%)'}

def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls)*label, new_cls)
    return new_cls



class LoveDA(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []
        if isinstance(image_dir, list) and isinstance(mask_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        elif isinstance(image_dir, list) and not isinstance(mask_dir, list):
            for img_dir_path in image_dir:
                self.batch_generate(img_dir_path, mask_dir)
        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms


    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))
        
        logger.info('%s -- Dataset images: %d' % (os.path.dirname(image_dir), len(rgb_filepath_list)))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        if len(self.cls_filepath_list) > 0:
            mask = imread(self.cls_filepath_list[idx]).astype(np.long) -1
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']
            
            return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']

            return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)


@er.registry.DATALOADER.register()
class LoveDALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = LoveDA(self.config.image_dir, self.config.mask_dir, self.config.transforms)
        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(LoveDALoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True,
                                       drop_last=True
                                       )
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))
