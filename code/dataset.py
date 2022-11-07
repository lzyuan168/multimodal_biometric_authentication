import torch
import cv2
import os
import utils
import numpy as np

from PIL import Image


class FingerprintDataset(torch.utils.data.Dataset):
    """
        Helper class to read the fingerprint dataset

        Parameters:
        -----------
        data_path: string
            string to the combined dataset folder
        mode: string
            must be one of 'train', 'test'
        transforms: object
            list of Transform objects composed using torchvision. 

        Output:
        -------
        image: numpy.ndarray
            the image from the dataset
        label: string
            the corresponding label for the image
    """

    def __init__(self, data_path, mode, folder, transforms=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.transforms = transforms
        self.folder = folder

        img_label_list = utils.get_img_label_list(data_path, mode, folder, aug=True)
        self.img_label_list = img_label_list


    def __len__(self):
        return len(self.img_label_list)


    def __getitem__(self, index):
        img_path = self.img_label_list[index][0]
        class_value = self.img_label_list[index][1]

        img = Image.open(img_path)
        #img = np.array(img)  #To be used for albumentations.transforms

        if self.transforms is not None:
            image = self.transforms(img)  #This is for torchvision.transforms

            #image = self.transforms(image=img)['image']  #This is for albumentations.transforms
            image = image.float()
        
        #label = torch.tensor(class_value)
        label = class_value

        return image, label


class ECGDataset(torch.utils.data.Dataset):
    """
        Helper class to read the ECG dataset

        Parameters:
        -----------
        data_path: string
            string to the combined dataset folder
        mode: string
            must be one of 'train', 'test', or 'ecg'
        transforms: object
            list of Transform objects composed using torchvision. 

        Output:
        -------
        image: numpy.ndarray
            the image from the dataset
        label: string
            the corresponding label for the image
    """

    def __init__(self, data_path, mode, transforms=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.transforms = transforms

        img_label_list = utils.get_ecg_img_label_list(data_path, mode)
        self.img_label_list = img_label_list


    def __len__(self):
        return len(self.img_label_list)


    def __getitem__(self, index):
        img_path = self.img_label_list[index][0]
        class_value = self.img_label_list[index][1]

        img = Image.open(img_path)
        #img = np.array(img)  #To be used for albumentations.transforms

        if self.transforms is not None:
            image = self.transforms(img)  #This is for torchvision.transforms

            #image = self.transforms(image=img)['image']  #This is for albumentations.transforms
            image = image.float()
        
        #label = torch.tensor(class_value)
        label = class_value

        return image, label


class FGECGDataset(torch.utils.data.Dataset):
    """
        Helper class to read the fingerprint and ECG dataset pairs

        Parameters:
        -----------
        data_path: string
            string to the combined dataset folder
        mode: string
            must be one of 'train', 'test'
        fg_transforms: object
            list of Transform objects composed using torchvision.
        ecg_transforms: object
            list of Transform objects composed using torchvision. 

        Output:
        -------
        image: numpy.ndarray
            the image from the dataset
        label: string
            the corresponding label for the image
    """

    def __init__(self, data_path, mode, fg_folder, ecg_folder, fg_transforms=None, ecg_transforms=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.fg_transforms = fg_transforms
        self.ecg_transforms = ecg_transforms
        self.fg_folder = fg_folder
        self.ecg_folder = ecg_folder

        img_label_list = utils.get_fg_ecg_img_label_list(data_path, mode, fg_folder, ecg_folder)
        self.img_label_list = img_label_list


    def __len__(self):
        return len(self.img_label_list)


    def __getitem__(self, index):
        fg_img_path = self.img_label_list[index][0]
        ecg_img_path = self.img_label_list[index][1]
        class_value = self.img_label_list[index][2]

        fg_img = Image.open(fg_img_path)
        ecg_img = Image.open(ecg_img_path)
        #img = np.array(img)  #To be used for albumentations.transforms

        
        fg_image = self.fg_transforms(fg_img)  #This is for torchvision.transforms
        ecg_image = self.ecg_transforms(ecg_img)

        #image = self.transforms(image=img)['image']  #This is for albumentations.transforms
        fg_image = fg_image.float()
        ecg_image = ecg_image.float()
        
        #label = torch.tensor(class_value)
        label = class_value

        return fg_image, ecg_image, label


class FGECGVerificationDataset(torch.utils.data.Dataset):
    """
        Helper class to read the fingerprint and ECG dataset pairs

        Parameters:
        -----------
        data_path: string
            string to the combined dataset folder
        mode: string
            must be one of 'train', 'test'
        fg_transforms: object
            list of Transform objects composed using torchvision.
        ecg_transforms: object
            list of Transform objects composed using torchvision. 

        Output:
        -------
        image: numpy.ndarray
            the image from the dataset
        label: string
            the corresponding label for the image
    """

    def __init__(self, data_path, mode, fg_folder, ecg_folder, fg_transforms=None, ecg_transforms=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.fg_transforms = fg_transforms
        self.ecg_transforms = ecg_transforms
        self.fg_folder = fg_folder
        self.ecg_folder = ecg_folder

        img_label_list = utils.get_fg_ecg_img_label_list(data_path, mode, fg_folder, ecg_folder)
        self.img_label_list = img_label_list


    def __len__(self):
        return len(self.img_label_list)


    def __getitem__(self, index):
        fg_img_path = self.img_label_list[index][0]
        ecg_img_path = self.img_label_list[index][1]
        class_value = self.img_label_list[index][2]

        fg_img = Image.open(fg_img_path)
        ecg_img = Image.open(ecg_img_path)
        #img = np.array(img)  #To be used for albumentations.transforms

        
        fg_image = self.fg_transforms(fg_img)  #This is for torchvision.transforms
        ecg_image = self.ecg_transforms(ecg_img)

        #image = self.transforms(image=img)['image']  #This is for albumentations.transforms
        fg_image = fg_image.float()
        ecg_image = ecg_image.float()
        
        #label = torch.tensor(class_value)
        if class_value <= 32:
            label = 0
        else:
            label = 1

        return fg_image, ecg_image, label


class DRIVEDBDataset(torch.utils.data.Dataset):
    """
        Helper class to read the DRIVEDB dataset

        Parameters:
        -----------
        data_path: string
            string to the combined dataset folder
        mode: string
            must be one of 'train', 'test'
        transforms: object
            list of Transform objects composed using torchvision. 

        Output:
        -------
        image: numpy.ndarray
            the image from the dataset
        label: string
            the corresponding label for the image
    """

    def __init__(self, data_path, mode, transforms=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.transforms = transforms

        img_label_list = utils.get_drivedb_img_label_list(data_path, mode)
        self.img_label_list = img_label_list


    def __len__(self):
        return len(self.img_label_list)


    def __getitem__(self, index):
        img_path = self.img_label_list[index][0]
        class_value = self.img_label_list[index][1]

        img = Image.open(img_path)
        #img = np.array(img)  #To be used for albumentations.transforms

        if self.transforms is not None:
            image = self.transforms(img)  #This is for torchvision.transforms

            #image = self.transforms(image=img)['image']  #This is for albumentations.transforms
            image = image.float()
        
        #label = torch.tensor(class_value)
        label = class_value

        return image, label