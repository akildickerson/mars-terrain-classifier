import torch
import numpy as np
from PIL import Image
from src.model.unet import MarsUNet

def run_inference(img_path, checkpoint, device=None):
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = MarsUNet().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    
    model.eval()
    image = Image.open(img_path).convert("RGB")
    image = image.resize((512, 512))
    image = torch.tensor(np.array(image).transpose(2, 0, 1)).float() / 255.0
    image = image.unsqueeze(0)
    image = image.to(device)

    return model.predict(image)
