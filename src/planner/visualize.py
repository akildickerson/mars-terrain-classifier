import numpy as np
from PIL import Image, ImageDraw

CLASS_COLORS = {
    0: (139, 90, 43),   # soil
    1: (128, 128, 128), # bedrock
    2: (210, 180, 140), # sand
    3: (178, 34, 34),   # large rock
    4: (34, 139, 34)    # rover track
}

def colorize_segmentation(seg_map):
    h, w = seg_map.shape
    arr = np.full((h, w, 3), 0)
    for idx, color in CLASS_COLORS.items():
        mask = seg_map == idx
        arr[mask] = color
    return arr.astype(np.uint8)

def draw_path(image, path):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for row, col in path:
        draw.ellipse([col-3, row-3, col+3, row+3], fill=(255, 255, 0))

    return np.array(image)
    