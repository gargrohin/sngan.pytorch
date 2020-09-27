import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.nn.functional as F
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
from torch.autograd import Variable

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths

fid_buffer_dir = os.path.join('../.', 'fid_buffer_biggan')
os.makedirs(fid_buffer_dir)

fname = '.samples.npz'
print('loading %s ...'%fname)
ims = np.load(fname)['x']
ims = list(ims.swapaxes(1,2).swapaxes(2,3))
mean, std = get_inception_score(ims)
print(mean,std)
fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
print(fid_score)

