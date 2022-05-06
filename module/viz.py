from cProfile import label
from dataclasses import replace
from PIL import Image
import numpy as np
import os
from data.dataset import COLOR_MAP, INT_TO_LABEL, LABEL_TO_HSL




def replace_labels( label_list, counts):
    new_label_list = []
    new_count_list = [0,0,0,0,0,0,0]
    new_label_list = list(INT_TO_LABEL.values())
    for i in range(7):
        if i in label_list:
            new_count_list[i] = counts[np.where(label_list == i)[0]]
    return new_label_list, new_count_list

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
        unique, counts = replace_labels(unique , counts)
        ratio_dict = dict(zip(unique, counts))
        total_count = np.sum(counts).item()
        ratio_dict = [{'id':k, 'label':k, 'value':round(int(v)/total_count,3),  "color": LABEL_TO_HSL[k] } for k, v in ratio_dict.items()]
        color_y = Image.fromarray(y_pred)
        color_y.putpalette(self.palette)
        color_y.save(os.path.join(self.out_dir, filename))
        return color_y, ratio_dict
