# Face Mask Recognition 

## Directory Overview

- `core/model`:
Contains all required model parameters.

## Usage

(NOTE: Not built for GPU)

- Pytorch installation:
    `$ pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`

- Then run `$ pip install -r requirements.txt`
- Then you can import and run `detect` function from `detector.py` file.
This function is meant to be run on inference


Arguments:
```
cv2_image            : (np.ndarray) Input cv2 image of type 
```

Returns:
```
'Mask' if mask is detected else 'No Mask'
```

## Sample Usage

Code:

```python
from core.detector import detect
import cv2

cv2_image = cv2.imread('image-path.png')

outputs = detect(cv2_image)
print(outputs)
```

Outputs:

```
Running initialization...

'Mask'
```



# Author

## __Rishik Mourya__

Contact for any query contact at __rishik@skylarklabs.ai__