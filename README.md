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
images              : Union[np.ndarray, List[np.ndarray]]
mask_threshold      : (float)
```

Returns:
If images is single np.ndarray
```
{
    'has_mask': [(x1, y1, x2, y2), (x1, y1, x2, y2), (x1, y1, x2, y2)],
    'has_no_mask': [(x1, y1, x2, y2), (x1, y1, x2, y2), (x1, y1, x2, y2)]
}
```
If images is list of np.ndarray
```
[
    {
        'has_mask': [(x1, y1, x2, y2), (x1, y1, x2, y2), (x1, y1, x2, y2)],
        'has_no_mask': [(x1, y1, x2, y2), (x1, y1, x2, y2), (x1, y1, x2, y2)]
    },
    {
        'has_mask': [(x1, y1, x2, y2), (x1, y1, x2, y2), (x1, y1, x2, y2)],
        'has_no_mask': [(x1, y1, x2, y2), (x1, y1, x2, y2), (x1, y1, x2, y2)]
    },
    ...
]
```

# Author

## __Rishik Mourya__

Contact for any query contact at __rishik@skylarklabs.ai__