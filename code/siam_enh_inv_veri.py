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


def main():

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
        

    # train_dataset = FGECGDataset(DATA_PATH, 'train', 'binary_inv', 'ecg_img', fg_transforms=fg_transforms, ecg_transforms=ecg_transforms)
    # test_dataset = FGECGDataset(DATA_PATH, 'test', 'binary_inv', 'ecg_img', fg_transforms=fg_transforms, ecg_transforms=ecg_transforms)

    train_dataset = FGECGVerificationDataset(DATA_PATH, 'train', 'enhanced_inv', 'ecg_img', fg_transforms=fg_transforms, ecg_transforms=ecg_transforms)
    test_dataset = FGECGVerificationDataset(DATA_PATH, 'test', 'enhanced_inv', 'ecg_img', fg_transforms=fg_transforms, ecg_transforms=ecg_transforms)


    #print(type(dataset), len(dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE
    )

    model = SiameseVGG16(fg_inchannel=1, ecg_inchannel=3).to(device)

    criterion = nn.CrossEntropyLoss()

    # optimizer with L2 Regularization with VGG16
    my_list = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))
    optimizer = optim.Adam([
                    {'params': [i[1] for i in params], 'lr': LR, 'weight_decay': 0.00001},
                    {'params': [i[1] for i in base_params], 'lr': LR}
    ])
    

    #optimizer = optim.Adam(model.parameters(), lr=LR)
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # softmax = nn.Softmax()

    # Weights and bias logging
    # wandb.login()
    wandb.init(
        project="Baseline_siam_experiment",
        entity="Zhiyuanthesis",
        name="siam_{}_run10(enh_inv_veri)_BN_L2".format(MODEL_NAME),
        config={
            "learning_rate": LR,
            "epochs": N_EPOCHS,
            "batch":BATCH_SIZE,
            "model": MODEL_NAME,
            "out_features": 65,
            "optimizer": "Adam"
        }
    )

    print("-"*100)
    print("--- Begin Training ---")
    print("-"*100)

    best_val_acc = 0

    model.train()
    for epoch in range(1, N_EPOCHS+1):
        train_loss = 0.0
        train_accuracy = 0.0
        train_samples = 0

        for fg_image, ecg_image, label in tqdm(train_loader):
            fg_image = fg_image.to(device)
            ecg_image = ecg_image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(fg_image, ecg_image)

            #output = softmax(output)

            loss = criterion(output, label)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            #accuracy = (output.argmax(dim=1) == label).float().mean()
            predictions = torch.argmax(output, dim=1)
            train_samples += predictions.size(0)
            
            #train_loss += loss
            train_loss += loss.item()
            #train_accuracy += accuracy
            train_accuracy += (predictions == label).sum()

        # Calculate avg. loss
        train_loss = train_loss / train_samples
        train_accuracy = train_accuracy / train_samples


        print("Validation ... ")

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

            print("Epoch ... {}/{} ---- train loss is : {} ---- train accuracy is : {}".format(epoch, N_EPOCHS, train_loss, train_accuracy))
            print("Epoch ... {}/{} ---- val loss is : {} ---- val accuracy is : {}".format(epoch, N_EPOCHS, val_loss, val_accuracy))

        if val_accuracy >= best_val_acc:
            # Save best checkpoint
            model_checkpoint_path = os.path.join(MODEL_CKPT, "{}{}_epoch_{}.pth".format(MODEL_NAME,"_siam_enh_inv_verification", epoch))
            torch.save( {"model_state_dict": model.state_dict()}, model_checkpoint_path)

            best_val_acc = val_accuracy


        wandb.log({
            "Epoch": epoch,
            "Batch":BATCH_SIZE,
            "Train_loss": train_loss,
            "Train_acc": train_accuracy,
            "Val_loss": val_loss,
            "Val_acc": val_accuracy
        })

if __name__ == "__main__":
    main()