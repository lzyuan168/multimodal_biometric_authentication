import os
import numpy as np
import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import wandb
import timm
import logging
import albumentations as A
import albumentations.pytorch
import time

from timm import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from imutils import paths
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import *


# General global variables
DATA_PATH = "../Datasets/combined"
MODEL_CKPT = "../models"
LOGS_PATH = "../logs"

# Model specific global variables
MODEL_NAME = "VGG16"
IMG_RESIZE = 256
IMG_CROPSIZE = 224  #224, 384
N_EPOCHS = 30
BATCH_SIZE = 32
LR = 0.00001


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
device = torch.device(device)

mdl = 'binfg_epoch_19_81percent.pth'
model = SiameseVGG16(fg_inchannel=1, ecg_inchannel=3).to(device)
ckpt_path = os.path.join(MODEL_CKPT, "VGG16_siam_{}".format(mdl))
print(ckpt_path)
checkpoint = torch.load(ckpt_path)
msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
print(msg)