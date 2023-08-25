import torch
import base64
import gradio as gr
import numpy as np
from PIL import Image
from io import BytesIO

MAX_COLORS = 12


def create_binary_matrix(img_arr, target_color):
    mask = np.all(img_arr == target_color, axis=-1)
    binary_matrix = mask.astype(int)
    return binary_matrix

def preprocess_mask(mask_, h, w, device):
    mask = np.array(mask_)
    mask = mask.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask

def process_sketch(canvas_data):
    binary_matrixes = []
    base64_img = canvas_data['image']
    image_data = base64.b64decode(base64_img.split(',')[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    im2arr = np.array(image)
    colors = [tuple(map(int, rgb[4:-1].split(','))) for rgb in canvas_data['colors']]
    colors_fixed = []
    
    r, g, b = 255, 255, 255
    binary_matrix = create_binary_matrix(im2arr, (r,g,b))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
    
    for color in colors:
        r, g, b = color
        if any(c != 255 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r,g,b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
            
    visibilities = []
    colors = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]
    
    return [gr.update(visible=True), binary_matrixes, *visibilities, *colors]

def process_prompts(binary_matrixes, *seg_prompts):
    return [gr.update(visible=True), gr.update(value=' , '.join(seg_prompts[:len(binary_matrixes)]))]
