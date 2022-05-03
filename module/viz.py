from PIL import Image
import numpy as np
import os


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
        ratio_dict = dict(zip(unique, counts))
        total_count = np.sum(counts).item() 
        ratio_dict = {k.item():v.item()/total_count for k, v in ratio_dict.items()}
        color_y = Image.fromarray(y_pred)
        color_y.putpalette(self.palette)
        color_y.save(os.path.join(self.out_dir, filename))
        return color_y, ratio_dict