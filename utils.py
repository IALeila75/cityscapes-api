from PIL import Image
import numpy as np

def apply_cityscapes_palette(mask):
    palette = [
        128, 64, 128, 244, 35, 232, 70, 70, 70,
        102, 102, 156, 190, 153, 153, 153, 153, 153,
        250, 170, 30, 220, 220, 0, 107, 142, 35,
        152, 251, 152, 70, 130, 180, 220, 20, 60,
        255, 0, 0, 0, 0, 142, 0, 0, 70,
        0, 60, 100, 0, 80, 100, 0, 0, 230
    ]
    palette += [0] * (768 - len(palette))  # Compléter à 256*3
    img_p = Image.fromarray(mask.astype(np.uint8), mode="P")
    img_p.putpalette(palette)
    return img_p.convert("RGB")
