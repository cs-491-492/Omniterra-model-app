import numpy as np
import PIL 

def add_transparent_layer(img, layer):
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    if layer.mode != 'RGBA':
        layer = layer.convert('RGBA')
        layer.resize(img.size)
    layer.putalpha(100)
    img_painted = PIL.Image.alpha_composite(img, layer)
    return img_painted