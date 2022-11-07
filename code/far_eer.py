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
from sklearn.metrics import roc_curve, RocCurveDisplay
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

log_path = os.path.join(LOGS_PATH, "far_eer.txt")

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


def far_eer():
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

    model_names = {'binary': ['binfg_verification_epoch_8_93percent.pth'],
                    'binary_inv': ['bin_inv_verification_fg_epoch_3_94percent.pth'],
                    'enhanced': ['enh_verification_epoch_2_94percent.pth'],
                    'enhanced_inv': ['enh_inv_verification_epoch_4_94percent.pth'],
                    'output': ['orifg_verification_epoch_27_97percent.pth']}

    criterion = nn.CrossEntropyLoss()

    for dt in dt_names:

        test_dataset_veri = FGECGVerificationDataset(DATA_PATH, 'test', dt, 'ecg_img', 
                                                fg_transforms=fg_transforms, ecg_transforms=ecg_transforms)

        
        for mdl in model_names[dt]:

            test_dataset = test_dataset_veri

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
            
            
            with torch.no_grad():
                
                model.eval()
                all_label = []
                all_score = []
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                
                for fg_image, ecg_image, label in tqdm(test_loader):
                    fg_image = fg_image.to(device)
                    ecg_image = ecg_image.to(device)
                    label = label.to(device)
                    output = model(fg_image, ecg_image)

                    # loss = criterion(output, label)

                    predictions = torch.argmax(output, dim=1)
                    print("predictions is : ", predictions)
                    print("label is : ", label)

                    probabilities = F.softmax(output, dim=1)[:, 1]
                    y_score = probabilities.cpu().numpy()

                    all_label.extend(label.cpu().numpy())
                    all_score.extend(y_score)

                    l = label.cpu().numpy()
                    p = predictions.cpu().numpy()

                    for i in range(len(l)):
                        if l[i] == 1:
                            if p[i] == 1:
                                TP += 1
                            else:
                                FN += 1
                        
                        else:
                            if p[i] == 0:
                                TN += 1
                            else:
                                FP += 1



                # 0 is imposter, 1 is genuine
                FAR = FP / (TN + FP)  # FP is number of imposter acceptance (GT=0, pred=1)
                FRR = FN / (TP + FN)  # FN is number of legitimate rejection ((GT=1, pred=0))

                log_file.write('TP, TN, FP, FN : {}, {}, {}, {} \n'.format(TP, TN, FP, FN))
                print('TP, TN, FP, FN : {}, {}, {}, {} \n'.format(TP, TN, FP, FN))
                log_file.write('FAR, FAR% : {}, {} \n'.format(FAR, FAR*100))
                print('FAR, FAR% : {}, {} \n'.format(FAR, FAR*100))
                log_file.write('FRR, FRR% : {}, {} \n'.format(FRR, FRR*100))
                print('FRR, FRR% : {}, {} \n'.format(FRR, FRR*100))


                fpr, tpr, thresholds = roc_curve(np.array(all_label), np.array(all_score))
                log_file.write('Thresholds : {} \n'.format(thresholds))
                print('Thresholds : {} \n'.format(thresholds))
                

                # Plotting
                # fig = plt.figure()
                # ax = fig.add_subplot(1,1,1)
                # plt.axis('off')
                plt.plot(fpr, tpr, marker='.')
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate' )
                # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                savedir = "../for_thesis/roc"
                plt.savefig(os.path.join(savedir, "roc_{}.jpg".format(mdl[:-4])))
                
                plot = RocCurveDisplay.from_predictions(np.array(all_label), np.array(all_score))
                plot.plot()
                plt.savefig(os.path.join(savedir, "roc_{}_sklearn.jpg".format(mdl[:-4])))
                print("{} ROC image saved".format(mdl))
                plt.close()
            
            log_file.write('-'*50 + '\n')

    
    log_file.close()


if __name__ == "__main__":
    far_eer()