if globals().get('_DINOV2_MATCHED_MEMBERS') != 20:
    raise RuntimeError('DINOv2 20/20 fingerprint gate did not pass')
rsna_phase('a5', 'START')
_A5_SAVED = dict(globals())
import gc, os, time, warnings
from concurrent.futures import ThreadPoolExecutor as A5DecodePool, as_completed
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
cv2.setNumThreads(1)
CROP_MM = 130.0
SIZE = 336
SLICE_BAND = (0.12, 0.88)
N_SLICE = 16
INTENSITY = 'slice'
SLOTS = [('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0)]
N_SLOT = len(SLOTS)
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

def _one_a5_root(tag, candidates, marker):
    hits = [Path(path) for path in candidates if Path(path).is_dir() and (Path(path) / marker).exists()]
    if len(hits) != 1:
        raise RuntimeError(f'{tag}: expected one direct root, found {hits}')
    return hits[0]

COMP = Path(ROOT)
CKPT = _one_a5_root('A5 weights', [
    '/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights',
    '/kaggle/input/knee-mri-fold-weights',
], 'm_f0.pt')
assert COMP is not None, 'competition data not attached'
assert CKPT is not None, 'fold weights not attached'
assert (COMP / 'sample_submission.csv').exists(), f'no competition data at {COMP}'
assert list(CKPT.glob('*_f*.pt')), f'no checkpoints at {CKPT}'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'