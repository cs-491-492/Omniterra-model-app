import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import os
from data.dataset import COLOR_MAP
from tqdm import tqdm
from module.tta import tta, Scale
from module.viz import VisualizeSegmm
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
import numpy as np
from skimage.io import imread
from albumentations import Compose, Normalize, Resize

import argparse

parser = argparse.ArgumentParser(description='Eval methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/deeplabv3p.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='baseline.deeplabv3p')
parser.add_argument('--img_path',  type=str,
                    help='img path', default='examples/exp')
parser.add_argument('--tta',  type=bool,
                    help='use tta', default=False)
args = parser.parse_args()

logger = logging.getLogger(__name__)

er.registry.register_all()

def evaluate(ckpt_path, config_path='base.hrnetw32', use_tta=False, img_path="examples/exp"):
    cfg = import_config(config_path)
    model_state_dict = torch.load(ckpt_path)

    log_dir = os.path.dirname(ckpt_path)
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    vis_dir = os.path.join(log_dir, 'vis-{}'.format(os.path.basename(ckpt_path)))
    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = VisualizeSegmm(vis_dir, palette)
    try:
        img = imread(img_path + '.png', pilmode='RGB')
    except:
        img = imread(img_path + '.jpeg', pilmode='RGB')
    
    transform = Compose([Resize(1024,1024),
                        Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                        er.preprocess.albu.ToTensor()
                ])
    blob = transform(image=img)
    img = blob['image']
    


    with torch.no_grad():
        img = img.cuda()

        
        #if img.shape[1] != 3:
        #    img = img[:,:3,:,:] 
        img = torch.unsqueeze(img, dim=0)
        if use_tta:
            pred = tta(model, img, tta_config=[
                Scale(scale_factor=0.5),
                Scale(scale_factor=0.75),
                Scale(scale_factor=1.0),
                Scale(scale_factor=1.25),
                Scale(scale_factor=1.5),
                Scale(scale_factor=1.75),
            ])
        else:
            pred = model(img)
        pred = pred.argmax(dim=1).cpu()

        for clsmap in pred:
            viz_op(clsmap.cpu().numpy().astype(np.uint8), 'exp_result.png')
    print('finished')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    evaluate(args.ckpt_path, args.config_path, args.tta, img_path=args.img_path)