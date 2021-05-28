from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import *
from PIL import Image
from .model import DISNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
transform = None
initialized = None

def initialize(reference_dir: str = None):
    global model, transform, initialized

    print('Running initialization...')

    model = DISNet()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model', 'parameters.pth')))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((136, 136)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
    ])

    initialized = True

@torch.no_grad()
def detect(
    cv2_image: np.ndarray,
):
    '''
    This function is meant to be run on inference.

    Arguments:
    cv2_image              : (np.ndarray) Input cv2 image of type 
    
    Returns:
        'Mask' if mask is detected else 'No Mask'
    '''

    global model, transform, initialized

    if not initialized:
        initialize()
    
    outputs = model(transform(Image.fromarray(cv2_image)).unsqueeze(0))

    probs = F.softmax(outputs)[0]

    label = f"Mask" if probs[0] >= 0.999 else f"No Mask"
    
    return label