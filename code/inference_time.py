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

log_path = os.path.join(LOGS_PATH, "inference_time.txt")

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


def inference_time():
    """
        This will calculate the inference time taken for each model.

        Parameters:
        -----------
        
    """

    log_file = open(log_path, 'w')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = ", device)
    device = torch.device(device)

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # for fingerprint
    fg_transforms = T.Compose([
        T.Resize((IMG_CROPSIZE, IMG_CROPSIZE)),        
        T.ToTensor(),        
        T.Normalize([0.5], [0.5])
    ])

    # for ecg
    ecg_transforms = T.Compose([
        T.Resize((IMG_RESIZE, IMG_RESIZE)),
        T.RandomCrop((IMG_CROPSIZE, IMG_CROPSIZE)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    dt_names = ['binary', 'binary_inv', 'enhanced', 'enhanced_inv', 'output']

    model_names = {'binary': ['binfg_epoch_19_81percent.pth', 'binfg_verification_epoch_8_93percent.pth'],
                    'binary_inv': ['bin_inv_fg_epoch_16_81percent.pth', 'bin_inv_verification_fg_epoch_3_94percent.pth'],
                    'enhanced': ['enh_fg_epoch_26_81percent.pth', 'enh_verification_epoch_2_94percent.pth'],
                    'enhanced_inv': ['enh_inv_fg_epoch_22_80percent.pth', 'enh_inv_verification_epoch_4_94percent.pth'],
                    'output': ['orifg_epoch_25_84percent.pth', 'orifg_verification_epoch_27_97percent.pth']}

    criterion = nn.CrossEntropyLoss()

    for dt in dt_names:

        test_dataset_nonveri = FGECGDataset(DATA_PATH, 'test', dt, 'ecg_img', 
                                                fg_transforms=fg_transforms, ecg_transforms=ecg_transforms)

        test_dataset_veri = FGECGVerificationDataset(DATA_PATH, 'test', dt, 'ecg_img', 
                                                fg_transforms=fg_transforms, ecg_transforms=ecg_transforms)

        
        for mdl in model_names[dt]:

            test_dataset = ''

            if 'verification' in mdl:
                test_dataset = test_dataset_veri
            else:
                test_dataset = test_dataset_nonveri


            log_file.write('Database name : {} \n'.format(dt))
            log_file.write('Model name : {} \n'.format(mdl))
            log_file.write('Length of test_dataset : {} \n'.format(len(test_dataset)))

            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=BATCH_SIZE
            )

            model = SiameseVGG16(fg_inchannel=1, ecg_inchannel=3).to(device)
            ckpt_path = os.path.join(MODEL_CKPT, "VGG16_siam_{}".format(mdl))
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            start_time = time.time()
            with torch.no_grad():
                val_loss = 0.0
                val_accuracy = 0.0
                val_samples = 0
                model.eval()
                
                for fg_image, ecg_image, label in tqdm(test_loader):
                    fg_image = fg_image.to(device)
                    ecg_image = ecg_image.to(device)
                    label = label.to(device)
                    output = model(fg_image, ecg_image)
                    #output = softmax(output)
                    loss = criterion(output, label)
                    #accuracy = (output.argmax(dim=1) == label).float().mean()
                    predictions = torch.argmax(output, dim=1)
                    print("predictions is : ", predictions)
                    print("label is : ", label)
                    val_accuracy += (predictions == label).sum()
                    val_samples += predictions.size(0)
                    val_loss += loss.item()
                    
                    #val_loss += loss
                    #val_accuracy += accuracy

                val_loss = val_loss / val_samples
                val_accuracy = val_accuracy / val_samples 
            
            end_time = time.time()
            elapsed_time = (end_time - start_time) / len(test_dataset)
            log_file.write('Elapsed time : {} \n'.format(elapsed_time))
            log_file.write('-'*50 + '\n')

    
    log_file.close()


if __name__ == "__main__":
    inference_time()