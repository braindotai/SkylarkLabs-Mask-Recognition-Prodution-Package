# 3K identities Masked Face Recognition 

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
reference_image        : (np.ndarray) Reference image in the form of numpy array read from cv2.imread 
                            (optional)

query_image            : (np.ndarray) Query image in the form of numpy array read from cv2.imread 
                            (optional)

reference_dir          : (str) Reference images directory.
                            (optional)

query_dir              : (str) Query images directory
                            (optional)
```

Returns:
```
if both reference_image and query_image is not None:
    returns {'similarity': <similarity reference_image and query_image>}
if both reference_dir and query_dir is not None:
    returns [{'reference_match': <best match reference path> 'similarity': <similarity reference_image and query_image>}, {}, ...]
else:
    RuntimeError
```

## Sample Usage

Code:

```python
from core.detector import detect
from pprint import pprint

outputs = detect(
    reference_dir = 'database',
    query_dir = 'test_images'
)
pprint(outputs)
```

Outputs:

```
Running initialization...
Computing database embeddings...

[{'query_image': 'test_images\\1.jpg',
  'reference_match': 'database\\face4.jpg',
  'similarity': 0.7702363729476929},
 {'query_image': 'test_images\\2.jpg',
  'reference_match': 'database\\face3.jpg',
  'similarity': 0.7404496073722839},
 {'query_image': 'test_images\\22.jpg',
  'reference_match': 'database\\sss1.png',
  'similarity': 0.8178597688674927},
 {'query_image': 'test_images\\amy-e1578040100498.jpg',
  'reference_match': 'database\\ss1.jpg',
  'similarity': 0.8497062921524048},
 {'query_image': 'test_images\\amy1.jpg',
  'reference_match': 'database\\ss1.jpg',
  'similarity': 0.8631435632705688},
 {'query_image': 'test_images\\face2.jpeg',
  'reference_match': 'database\\face1.jpeg',
  'similarity': 0.7775507569313049},
 {'query_image': 'test_images\\face5.jpg',
  'reference_match': 'database\\face4.png',
  'similarity': 0.8321362733840942},
 {'query_image': 'test_images\\isla2.jpg',
  'reference_match': 'database\\ss2.jpg',
  'similarity': 0.8287513256072998},
 {'query_image': 'test_images\\Jeffrey-Dean-Morgan-Wallpapers-HD.jpg',
  'reference_match': 'database\\s1.png',
  'similarity': 0.846564769744873},
 {'query_image': 'test_images\\lJavier Bardem.jpg',
  'reference_match': 'database\\s2.png',
  'similarity': 0.7938718795776367},
 {'query_image': 'test_images\\lv.jpg',
  'reference_match': 'database\\sss2.png',
  'similarity': 0.7832841873168945},
 {'query_image': 'test_images\\me1.jpeg',
  'reference_match': 'database\\face4.png',
  'similarity': 0.8676998019218445},
 {'query_image': 'test_images\\me2.jpeg',
  'reference_match': 'database\\face4.png',
  'similarity': 0.8796103596687317},
 {'query_image': 'test_images\\r2.png',
  'reference_match': 'database\\r3.png',
  'similarity': 0.7854234576225281},
 {'query_image': 'test_images\\ss2.jpg',
  'reference_match': 'database\\ss2.jpg',
  'similarity': 0.7756043076515198},
 {'query_image': 'test_images\\withoutglasses.jpg',
  'reference_match': 'database\\glasses.jpg',
  'similarity': 0.7450568675994873},
 {'query_image': 'test_images\\young_amit.jpg',
  'reference_match': 'database\\face4.jpg',
  'similarity': 0.8112750053405762}]  
```



# Author

## __Rishik Mourya__

Contact for any query contact at __rishik@skylarklabs.ai__