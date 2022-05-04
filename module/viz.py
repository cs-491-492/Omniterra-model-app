from cProfile import label
from dataclasses import replace
from PIL import Image
import numpy as np
import os

COLOR_MAP = {0:'Background', 1:'Building', 2:'Road', 3:'Water', 4:'Barren', 5:'Forest', 6:'Agricultural'}

def replace_labels( label_list ):
    new_label_list = []
    for i in range(len(label_list)):
        new_label_list.append(COLOR_MAP[label_list[i]])
    return new_label_list

class VisualizeSegmm(object):
    def __init__(self, out_dir, palette):
        self.out_dir = out_dir
        self.palette = palette
        os.makedirs(self.out_dir, exist_ok=True)

    def __call__(self, y_pred, filename):
        """
        Args:
            y_pred: 2-D or 3-D array of shape [1 (optional), H, W]
            filename: str
        Returns:
        """
        y_pred = y_pred.astype(np.uint8)
        y_pred = y_pred.squeeze()
        unique, counts = np.unique(y_pred, return_counts=True)
        print(unique)
        unique = replace_labels(unique)
        ratio_dict = dict(zip(unique, counts))
        total_count = np.sum(counts).item() 
        ratio_dict = [{'x':k, 'y':v.item()/total_count} for k, v in ratio_dict.items()]
        color_y = Image.fromarray(y_pred)
        color_y.putpalette(self.palette)
        color_y.save(os.path.join(self.out_dir, filename))
        return color_y, ratio_dict
