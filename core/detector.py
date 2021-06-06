from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
from typing import *
from .face_detection_model import FaceDetection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMETER_PATH = os.path.join(BASE_DIR, 'model', 'parameters.onnx')

model = None
initialized = None

def initialize():
    global model, transform, initialized

    print('Running initialization...')

    model = FaceDetection(PARAMETER_PATH, faceThreshold = 0.725)

    initialized = True

@torch.no_grad()
def detect(
    images: Union[np.ndarray, List[np.ndarray]] = None,
    mask_threshold: float = 0.9,
):
    '''
    This function is meant to be run on inference.

    Arguments:
    images              : (np.ndarray)
    mask_threshold      : (float)

    Returns:
        'Mask' if mask is detected else 'No Mask'
    '''

    global model, initialized

    if not initialized:
        initialize()
    
    if isinstance(images, np.ndarray):
        result = {
            'has_mask': [],
            'has_no_mask': []
        }

        outputFaces = model.detect(images)

        for faceIdx in range(outputFaces.shape[0]):
            faceBox = outputFaces[faceIdx, 0: 4]
            maskProb = outputFaces[faceIdx, 5]

            result['has_mask' if maskProb > mask_threshold else 'has_no_mask'].append((
                int(faceBox[0]), int(faceBox[1]), int(faceBox[2]), int(faceBox[3])
            ))

        return result

    elif isinstance(images, list):
        return [detect(image) for image in images]