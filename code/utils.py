import os
import glob
import shutil
import pathlib
import random
import cv2
import Augmentor
#import biosppy
import wfdb
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import albumentations as A

from tqdm import tqdm
from PIL import Image
from skimage.morphology import skeletonize, thin
from skimage.util import invert
from skimage.io import imread


ecg_dataset_path = "../Datasets/ECG/CYBHi/data/short-term"
drivedb_dataset_path = "../Datasets/ECG/DRIVEDB/stress-recognition-in-automobile-drivers-1.0.0"
data_path = "../Datasets/ECG/DRIVEDB/processed"
fingerprint02_dataset_path = "../Datasets/Fingerprint/FVC2002"
fingerprint04_dataset_path = "../Datasets/Fingerprint/FVC2004"
combined_dataset_path = "../Datasets/combined"
#finger_aug_path = "../Datasets/Fingerprint/FVC2002_aug"


def copy_ecg_to_combined_dataset(ecg_path, combined_path):
    """
        This will copy the ECG files from the original folder to the combined folder which
        contains both the ECG and Fingerprint data paired to form individual subject.

        Parameters
        ----------
        ecg_path : string
            path to original ecg signal files folders
        combined_path : string
            path to the destination folder
    """

    # Copy the ECG subject files to new folder
    count = 1
    for files in tqdm(glob.glob(ecg_path + '/*')):
        ecg_mode = os.path.basename(files).split('-')[2]
        device_code = os.path.basename(files).split('-')[3]
        filename = os.path.basename(files)
        dest_path = os.path.join(combined_path, 'S{}'.format(count))

        if ecg_mode == 'CI' and device_code == '8B.txt':
            
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            sbj_path = os.path.join(dest_path, filename)
            print(filename)
            print(sbj_path)
            shutil.copy2(files, sbj_path)
            print(count)
            count += 1            
            
    print('Finished copying files')

#print(copy_ecg_to_combined_dataset(ecg_dataset_path, combined_dataset_path))


def copy_fingerprint_to_combined_dataset(finger1_path, finger2_path, combined_path):
    """
        This will copy the Fingerprint images from the original folder to the combined folder which
        contains both the ECG and Fingerprint data paired to form individual subject.

        Parameters:
        -----------
        finger1_path, finger2_path : string
            path to 2 different fingerprint folders
        combined_path : string
            path to the destination folder
    """

    # Copy the FVC2002 images to new folder
    count = 1
    img_count = 1
    for paths in [finger1_path, finger2_path]:
        dataset_name = os.path.basename(paths)

        for dbs in sorted(tqdm(glob.glob(paths + '/*'))):
            db_name = os.path.basename(dbs)

            if os.path.isdir(dbs) and db_name != 'DB4':            
                print(db_name)

                for imgs in sorted(tqdm(glob.glob(dbs + '/*'))):
                    print(imgs)
                    img_name = os.path.basename(imgs)
                    image_name = '{}_{}_{}'.format(dataset_name, db_name, img_name)
                    print(image_name)
                    dest_path = os.path.join(combined_path, 'S{}'.format(count))
                    
                    if img_count < 9:
                        if not os.path.exists(dest_path):
                            os.makedirs(dest_path)

                        sbj_path = os.path.join(dest_path, image_name)
                        print(sbj_path)
                        shutil.copy2(imgs, sbj_path)
                        img_count += 1
                    else:
                        count += 1
                        dest_path = os.path.join(combined_path, 'S{}'.format(count))
                        
                        if not os.path.exists(dest_path):
                            os.makedirs(dest_path)

                        sbj_path = os.path.join(dest_path, image_name)
                        print(sbj_path)
                        shutil.copy2(imgs, sbj_path)
                        img_count = 2
                        print(count)
        
    print('Finished copying files')

#print(copy_fingerprint_to_combined_dataset(fingerprint02_dataset_path, fingerprint04_dataset_path, combined_dataset_path))


def split_ecg_train_test_data(combined_path):
    """
        The origianl ECG signal file will be split into 70% training and 30% testing
        and will be stored in different files.

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        print(sbj)
        for files in tqdm(glob.glob(sbj + '/*')):
            file_extension = pathlib.Path(files).suffix
            filename = os.path.basename(files)
            train_filename = "{}_train.txt".format(filename[:-4])
            test_filename = "{}_test.txt".format(filename[:-4])
            
            if file_extension == '.txt':
                print(filename, file_extension)

                with open(files, 'r') as f:
                    lines = f.readlines()
                    total_lines = len(lines)

                print(filename, total_lines)
                train_lines = (total_lines - 8) * 70 // 100
                print(train_lines)
                
                for line in lines:
                    line_list = line.split()

                    if line_list[0] == '#':
                        continue
                    elif line_list[0] != '#' and train_lines != 0:
                        train_path = os.path.join(sbj, train_filename)
                        print(train_path)
                        with open(train_path, 'a+') as af:
                            af.write(line_list[3] + "\n")
                        train_lines -= 1
                    elif line_list[0] != '#' and train_lines == 0:
                        test_path = os.path.join(sbj, test_filename)
                        print(test_path)
                        with open(test_path, 'a+') as taf:
                            taf.write(line_list[3] + "\n")
    
    print("Train Test Split completed")                

#print(split_ecg_train_test_data(combined_dataset_path))


def extract_ecg_readings(combined_path):
    """
        The origianl ECG signal file will be will be read and only the column for ECG
        will be extracted and stored in a different file.

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        print(sbj)
        for files in tqdm(glob.glob(sbj + '/*')):
            file_extension = pathlib.Path(files).suffix
            filename = os.path.basename(files)
            ecg_filename = "{}_ecg.txt".format(filename[:-4])
            
            if file_extension == '.txt':
                if "train" in filename or "test" in filename:
                    pass
                else:
                    with open(files, 'r') as f:
                        lines = f.readlines()
                        total_lines = len(lines)

                    print(filename, total_lines)
                    
                    for line in lines:
                        line_list = line.split()

                        if line_list[0] == '#':
                            continue
                        else:
                            ecg_path = os.path.join(sbj, ecg_filename)
                            print(ecg_path)
                            with open(ecg_path, 'a+') as af:
                                af.write(line_list[3] + "\n")
    
    print("Extracting ECG readings completed")                

#print(extract_ecg_readings(combined_dataset_path))


def split_fingerprint_train_test_data(combined_path):
    """
        The 8 fingerprint images will be split into first 4 for training and last 4 for testing,
        and will be named accordingly.

        Parameters:
        -----------
        combined_path : string  
            path to the combined data folder containing individual subject folders
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        img_count = 1
        for files in sorted(tqdm(glob.glob(sbj + '/*'))):
            file_extension = pathlib.Path(files).suffix
            filename = os.path.basename(files)
            train_filename = "{}_train.tif".format(filename[:-4])
            test_filename = "{}_test.tif".format(filename[:-4])

            if file_extension == '.tif':
                print(filename, file_extension)

                if img_count < 5:
                    ori_path = os.path.join(sbj, filename)
                    re_path = os.path.join(sbj, train_filename)
                    os.rename(ori_path, re_path)
                    img_count += 1
                    print(img_count)
                else:
                    print(img_count)
                    ori_path = os.path.join(sbj, filename)
                    re_path = os.path.join(sbj, test_filename)
                    os.rename(ori_path, re_path)
                    img_count += 1                    
        
    print('Train Test Split completed')

#print(split_fingerprint_train_test_data(combined_dataset_path))


def convert_ecg_signal_to_image(combined_path, mode):
    """
        This will take the train and test ECG signal files and convert them to ECG images respectively.
        This will first perform segmentation on raw ECG signals into individual beats, then
        generates images from the segmented beats.

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
        mode : string
            defines whether "train" or "test" or 
            if "ecg" then its the unsplit data
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        for files in tqdm(glob.glob(sbj + '/*')):
            file_extension = pathlib.Path(files).suffix
            filename = os.path.basename(files)

            if file_extension == '.txt':
                modename = os.path.basename(files).split('_')[-1]

                if modename == "{}.txt".format(mode):
                    print(filename)
                    data = np.genfromtxt(files, dtype=None, encoding=None)
                    print(type(data), len(data), data)

                    signals = []
                    count = 1
                    # get separate waves
                    peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=1000)[0]
                    print("length of peaks : {}".format(len(peaks)))
                    
                    for i in (peaks[1:-1]):
                        diff1 = abs(peaks[count - 1] - i)
                        diff2 = abs(peaks[count + 1] - i)
                        x = peaks[count - 1] + diff1 // 2
                        y = peaks[count + 1] - diff2 // 2
                        signal = data[x:y]
                        signals.append(signal)
                        count += 1
                        #print(count)
                    print("signals finished")

                    # save the waves to images
                    ecg_count = 1
                    savedir = os.path.join(sbj, "ecg_img", mode)    
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)

                    print("length of signals : {}".format(len(signals)))
                    for sig in signals:
                        fig = plt.figure()
                        ax = fig.add_subplot(1,1,1)
                        plt.axis('off')
                        plt.plot(sig)

                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        plt.savefig(os.path.join(savedir, "ecg_{}.jpg".format(ecg_count)), bbox_inches=extent)
                        print("{} ECG image {} saved".format(os.path.basename(sbj), ecg_count))
                        plt.close()
                        ecg_count += 1

#print(convert_ecg_signal_to_image(combined_dataset_path, "test"))

def for_thesis_paper():

    files = "../Datasets/combined/S1/20110719-VRR-CI-8B_train.txt"
    data = np.genfromtxt(files, dtype=None, encoding=None)
    peaks = biosppy.signals.ecg.ssf_segmenter(signal=data, sampling_rate=1000)[0]

    signals = []
    count = 1
    
    for i in (peaks[1:-1]):
        diff1 = abs(peaks[count - 1] - i)
        diff2 = abs(peaks[count + 1] - i)
        x = peaks[count - 1] + diff1 // 2
        y = peaks[count + 1] - diff2 // 2
        signal = data[x:y]
        signals.append(signal)
        count += 1

    # save the waves to images
    ecg_count = 1
    savedir = "../for_thesis/ssf"  
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for sig in signals:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.plot(sig)

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(os.path.join(savedir, "ecg_{}.jpg".format(ecg_count)), bbox_inches=extent)
        plt.close()
        ecg_count += 1

#print(for_thesis_paper())

def convert_ecg_Rpeaks_to_img(combined_path, mode):
    """
        This will take the raw ECG signal and find the R-peaks, then exclude 20 readings
        after the previous R-peak and 20 readings before the next R-peak and 
        convert them to ECG images respectively.

        This will first perform segmentation on raw ECG signals into individual beats, then
        generates images from the segmented beats.

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
        mode : string
            defines whether "train" or "test" or 
            if "ecg" then its the unsplit data
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        for files in tqdm(glob.glob(sbj + '/*')):
            file_extension = pathlib.Path(files).suffix
            filename = os.path.basename(files)

            if file_extension == '.txt':
                modename = os.path.basename(files).split('_')[-1]

                if modename == "{}.txt".format(mode):
                    print(filename)
                    data = np.genfromtxt(files, dtype=None, encoding=None)
                    #print(type(data), len(data), data)

                    rpeaks = biosppy.signals.ecg.ecg(signal=data, sampling_rate=1000, show=False)['rpeaks']
                    # heartbeats = biosppy.signals.ecg.ecg(signal=data, sampling_rate=1000, show=False)['templates']
                    heartbeats = biosppy.signals.ecg.extract_heartbeats(signal=data, rpeaks=rpeaks, sampling_rate=1000, before=0.2, after=0.2)[0]

                    ecg_count = 1
                    savedir = os.path.join(sbj, "ecg_img", mode)    
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)

                    for sig in heartbeats:
                        fig = plt.figure()
                        ax = fig.add_subplot(1,1,1)
                        plt.axis('off')
                        plt.plot(sig)

                        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        plt.savefig(os.path.join(savedir, "ecg_{}.jpg".format(ecg_count)), bbox_inches=extent)
                        print("{} ECG image {} saved".format(os.path.basename(sbj), ecg_count))
                        plt.close()
                        ecg_count += 1

#print(convert_ecg_Rpeaks_to_img(combined_dataset_path, 'ecg'))


def fingerprint_enhance(combined_path):
    """
        This will enhance the fingerprint images through image enhancement, binarization 
        and thinning.

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        for files in tqdm(glob.glob(sbj + '/*')):
            foldername = os.path.basename(files)
            if foldername == 'output':
                for each in tqdm(glob.glob(files + '/*')):
                    img_path = each
                    img_name = os.path.basename(each)
                    print(img_name)
                    img = Image.open(img_path)
                    img = np.array(img)
                    # cv2.imshow('original', img)
                    # cv2.waitKey(0)

                    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    clahe_img = clahe.apply(img)
                    print("CLAHE done")
                    # cv2.imshow('CLAHE', clahe_img)
                    # cv2.waitKey(0)

                    # Apply Otsu adaptive binarization
                    (T, bin_img_otsu) = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    #print(type(bin_img_otsu))
                    print("Binarization done")
                    # cv2.imshow('bin_img_otsu', bin_img_otsu)
                    # cv2.waitKey(0)

                    # Apply image thinning
                    thin_img = skeletonize((invert(bin_img_otsu) / bin_img_otsu.max()))
                    print("Thinning done")
                    # plt.imshow(thin_img)
                    # plt.show()

                    # Save image
                    save_dir = os.path.join(sbj, 'enhanced')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, img_name[:-4]+'.jpg')
                    print(save_path)
                    thin_img = Image.fromarray(invert(thin_img)) # invert controls black&white or white&black
                    thin_img.save(save_path)
                    print("Image saved")

#print(fingerprint_enhance(combined_dataset_path))


def get_subj_name_list(combined_path):
    """
        This will print a list of all the subject names, i.e., this will be the label

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders

        Output:
        -------
        subj_list : list
            list of string value containing subject name, the list will be the label
    """

    subj_list = []
    for sbj in tqdm(glob.glob(combined_path + '/*')):
        print(sbj)
        sbj_name = os.path.basename(sbj)
        subj_list.append(sbj_name)

    return sorted(subj_list)

#print(get_subj_name_list(combined_dataset_path))


def get_img_label_list(combined_path, mode, folder_name, aug=False):
    """
        Gets a list of lists, which contains [img_path, label] for each list

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
        mode : string
            defines whether "train" or "test"
        folder_name : string
            defines whether to use enhanced or enhanced_inv images
        aug : boolean
            if aug set to True, then use the augmented images

        Output:
        -------
        img_label_list : list
            list of lists, which contains [img_path, label] for each list
    """

    img_label_list = []

    if aug:
        for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
            label = int(os.path.basename(sbj)[1:])-1
            for folder in tqdm(glob.glob(sbj + '/*')):
                if os.path.isdir(folder):
                    foldername = os.path.basename(folder)
                    if foldername == folder_name:            
                        for files in sorted(tqdm(glob.glob(folder + '/*'))):
                            modename = os.path.basename(files).split('_')[6]

                            if modename == "{}.tif".format(mode):
                                img_label_list.append([files, label])
    else:
        for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
            label = int(os.path.basename(sbj)[1:])-1
            for files in sorted(tqdm(glob.glob(sbj + '/*'))):
                file_extension = pathlib.Path(files).suffix

                if file_extension == '.tif':
                    modename = os.path.basename(files).split('_')[4]

                    if modename == "{}{}".format(mode, file_extension):
                        img_label_list.append([files, label])
    
    return img_label_list


#print(get_img_label_list(combined_dataset_path, "train", "enhanced", aug=True))


def get_ecg_img_label_list(combined_path, mode):
    """
        Gets a list of lists, which contains [img_path, label] for each list

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
        mode : string
            defines whether "train" or "test" 

        Output:
        -------
        img_label_list : list
            list of lists, which contains [img_path, label] for each list
    """

    img_label_list = []

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        label = int(os.path.basename(sbj)[1:])-1
        img_path = os.path.join(sbj, 'ecg_img', mode)
        for files in sorted(tqdm(glob.glob(img_path + '/*'))):
            print("file path is : {}".format(files))
            img_label_list.append([files, label])
    
    return img_label_list


#print(get_ecg_img_label_list(combined_dataset_path, "train"))


def get_fg_ecg_img_label_list(combined_path, mode, fg_folder_name, ecg_folder_name):
    """
        Gets a list of lists, which contains [img_path, label] for each fingerprint, ECG pair

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
        mode : string
            defines whether "train" or "test"
        fg_folder_name : string
            defines whether to use enhanced / enhanced_inv / binary / binary_inv images
        ecg_folder_name : string
            defines folder name of ECG
        

        Output:
        -------
        img_label_list : list
            list of lists, which contains [fg_img_path, ecg_img_path, label] for each list
    """

    img_label_list = []

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        fg_list = []
        ecg_list = []
        label = int(os.path.basename(sbj)[1:])-1
        for folder in tqdm(glob.glob(sbj + '/*')):
            if os.path.isdir(folder):
                foldername = os.path.basename(folder)
                # getting fingerprint images
                if foldername == fg_folder_name:            
                    for fg_files in sorted(tqdm(glob.glob(folder + '/*'))):
                        modename = os.path.basename(fg_files).split('_')[6]

                        if modename == "{}.tif".format(mode):
                            #print("fingerprint file path is : {}".format(fg_files))
                            fg_list.append(fg_files)

                # getting fingerprint images
                if foldername == ecg_folder_name:
                    folder_path = os.path.join(folder, mode)
                    for ecg_files in sorted(tqdm(glob.glob(folder_path + '/*'))):
                        #print("ECG file path is : {}".format(ecg_files))
                        ecg_list.append(ecg_files)

        # combining the ECG, fingerprint pairs
        for i in range(len(fg_list)):
            fg = fg_list[i]
            ecg = random.choice(ecg_list)
            img_label_list.append([fg, ecg, label])

    return img_label_list

#print(get_fg_ecg_img_label_list(combined_dataset_path, "train", "binary", "ecg_img"))



def swap_train_test_data(combined_path, num_to_swap, mode):
    """
        This will select num_to_swap number of train/test data and swap their mode.

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
        num_to_swap : int
            number of data to swap
        mode : string
            defines whether "train" or "test"
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        img_count = 1
        for files in sorted(tqdm(glob.glob(sbj + '/*'))):
            file_extension = pathlib.Path(files).suffix
            filename = os.path.basename(files)

            if file_extension == '.tif':
                print(filename)

                if filename.split('_')[-1] == "{}{}".format(mode, file_extension):
                    if img_count <= num_to_swap:
                        if mode == 'train':
                            swap = 'test'
                            swap_name = "{}{}{}".format(filename[:-9], swap, file_extension)
                            ori_path = os.path.join(sbj, filename)
                            re_path = os.path.join(sbj, swap_name)
                            os.rename(ori_path, re_path)
                            img_count += 1
                            print(img_count)
                        else:
                            swap = 'train'
                            swap_name = "{}{}{}".format(filename[:-8], swap, file_extension)
                            ori_path = os.path.join(sbj, filename)
                            re_path = os.path.join(sbj, swap_name)
                            os.rename(ori_path, re_path)
                            img_count += 1
                            print(img_count)

#print(swap_train_test_data(combined_dataset_path, 2, 'test'))


def fingerprint_data_augmentations(combined_path):
    """
        This method will augment the original FVC2002 and FVC2004 images through RandomCrop, Rotate, RandomBrightness, RandomContrast
        It will first resize the original image to 256x256, then apply the transforms

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        aug = Augmentor.Pipeline(sbj)
        aug.resize(probability=1.0, width=256, height=256, resample_filter="BILINEAR")
        aug.crop_by_size(probability=1.0, width=224, height=224, centre=False)  #center=False, will crop random regions
        aug.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)  #It has a limit of 25 degrees
        aug.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
        aug.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.0)
        #aug.random_erasing()  # An operation to add noise by adding random rectangle area of random pixel value

        num_of_samples = 8000
        aug.sample(num_of_samples)

#print(fingerprint_data_augmentations(combined_dataset_path))

def aug_for_thesis():

    file_path = "../for_thesis/fg"

    aug = Augmentor.Pipeline(file_path)
    aug.resize(probability=1.0, width=256, height=256, resample_filter="BILINEAR")
    aug.crop_by_size(probability=1.0, width=224, height=224, centre=False)
    aug.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)
    aug.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
    aug.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.0)

    num_of_samples = 50
    aug.sample(num_of_samples)

    print("Done")

#print(aug_for_thesis())


def delete_folders(combined_path, folder_name):
    """
        This method will remove unneeded folders

        Parameters:
        -----------
        combined_path : string
            path to the combined data folder containing individual subject folders
        folder_name : string
            name of the folder to be removed
    """

    for sbj in sorted(tqdm(glob.glob(combined_path + '/*'))):
        for folder in tqdm(glob.glob(sbj + '/*')):
            if os.path.isdir(folder):
                foldername = os.path.basename(folder)

                if foldername == folder_name:
                    shutil.rmtree(folder)

#print(delete_folders(combined_dataset_path, "ecg_img"))


def process_drivedb(drivedb_path):
    """
        This method will read the driver stress detection dataset and 
        split to train and test sets and 
        save to combined dataset folder

        Parameters:
        -----------
        drivedb_path : string
            path to the drivedb dataset folder
        combined_path : string
            path to the combined data folder containing individual subject folders
    """

    # Reading .dat files for all 17 drivers
    def read_signals_and_meta():
        
        all_signals = []
        meta_data = []

        for driver in sorted(tqdm(glob.glob(drivedb_path + '/*'))):
            if driver.endswith('.dat'):
                print(driver)
                signals, fields = wfdb.rdsamp(driver[:-4])

                all_signals.append(signals)
                meta_data.append(fields)

        return all_signals, meta_data

    def display_indv_driver_info(meta_data):
        
        i = 1
        for item in meta_data:
            print("driver",i)
            i=i+1
            print("total time(min):", round(item['sig_len']/15.5/60, 2))
            print("sigal number:",item['n_sig'])
            print("sigal name:",item['sig_name'])
            print("--------------------")

    def save_indv_driver_ecg(all_signals, meta_data):

        save_dir = "../Datasets/ECG/DRIVEDB/processed"
        for i  in range(1, len(all_signals)+1):
            driver_idx = "driver_{}".format(i)
            ecg_data = all_signals[i-1][:,0]
            # print(ecg_data)
            # print(len(ecg_data))
            # print('meta : ', meta_data[i]['sig_len'])
            # print(all_signals[i])
            save_file = os.path.join(save_dir, "{}.txt".format(driver_idx))
            with open(save_file, 'a+') as af:
                for each in ecg_data:
                    af.write("{}\n".format(each))
            print("Finished saving for driver {}".format(i))

    def extracting_starting_time_df():

        df_time_intervals = pd.read_csv(os.path.join(drivedb_path, 'marker_info.csv'))
        #print(df_time_intervals)

        # Renaming column for convenience
        df_time_intervals.columns = ['Drive', 'Rest1', 'City1', 'Hwy1', 'City2', 'Hwy2', 'City3', 'Rest2', 'Total']
        # Getting starting time for each activity
        df_time_start = df_time_intervals.copy()
        df_time_start['Rest1'] = 0
        df_time_start['City1'] = df_time_start['Rest1'] + df_time_intervals['Rest1']
        df_time_start['Hwy1'] = df_time_start['City1'] + df_time_intervals['City1']
        df_time_start['City2'] = df_time_start['Hwy1'] + df_time_intervals['Hwy1']
        df_time_start['Hwy2'] = df_time_start['City2'] + df_time_intervals['City2']
        df_time_start['City3'] = df_time_start['Hwy2'] + df_time_intervals['Hwy2']
        df_time_start['Rest2'] = df_time_start['City3'] + df_time_intervals['City3']

        # Calculating the start time for sampling，usually after 5 min for each activity (mentioned in paper)
        #df_time_sample = df_time_start.drop(['Drive', 'Total'], axis=1)
        df_time_sample = df_time_start
        # Rest1 sampling started after 10 min
        df_time_sample['Rest1'] += 10
        # City1 sampling started after 5 min
        df_time_sample['City1'] += 5
        # Hwy1 sampling started after 4 min
        df_time_sample['Hwy1'] += 4
        # City2 sampling started after 3 min
        df_time_sample['City2'] += 3
        # Hwy2 sampling started after 4 min
        df_time_sample['Hwy2'] += 4
        # City3 sampling started after 5 min
        df_time_sample['City3'] += 5
        # Rest2 sampling started after 5 min
        df_time_sample['Rest2'] += 5
        # Driver09 and Driver16 does not have Rest2 timing，dropping the rows
        df_time_sample.loc[4, 'Rest2'] = None
        df_time_sample.loc[9, 'Rest2'] = None
        new_df = df_time_sample.drop(labels=[4, 9], axis=0)

        # print(df_time_sample)
        # print(new_df)
        return new_df

    def save_train_test_per_driver(new_df):

        file_dir = "../Datasets/ECG/DRIVEDB/processed"
        for i in range(new_df.shape[0]):
            driver_info = new_df.iloc[i, 0]
            rest1_start = new_df.iloc[i, 1]
            city1_start = new_df.iloc[i, 2]
            hwy1_start = new_df.iloc[i, 3]
            city3_start = new_df.iloc[i, 6]
            rest2_start = new_df.iloc[i, 7]
            total = new_df.iloc[i, 8]

            # rest1_duration = city1_start - rest1_start
            # city1_duration = hwy1_start - city1_start
            # city3_duration = rest2_start - city3_start
            # rest2_duration = total - rest2_start

            # print(driver_info, rest1_start, city1_start, hwy1_start, city3_start, rest2_start, total)
            # print(rest1_duration, city1_duration, city3_duration, rest2_duration)

            file_name = os.path.join(file_dir, "{}.txt".format(driver_info))
            with open(file_name, 'r') as f:
                lines = f.readlines()
                total_lines = len(lines)
            
            train_ns_sig_start_idx = int(15.5 * 60 * rest1_start)
            train_ns_sig_stop_idx = int(15.5 * 60 * city1_start)

            train_s_sig_start_idx = int(15.5 * 60 * city1_start)
            train_s_sig_stop_idx = int(15.5 * 60 * hwy1_start)

            test_s_sig_start_idx = int(15.5 * 60 * city3_start)
            test_s_sig_stop_idx = int(15.5 * 60 * rest2_start)

            test_ns_sig_start_idx = int(15.5 * 60 * rest2_start)
            test_ns_sig_stop_idx = int(15.5 * 60 * total)

            save_file_train_ns = os.path.join(file_dir, "{}_ns_train.txt".format(driver_info))
            with open(save_file_train_ns, 'a+') as af:
                for each in lines[train_ns_sig_start_idx:train_ns_sig_stop_idx]:
                    af.write("{}\n".format(each))

            save_file_train_s = os.path.join(file_dir, "{}_s_train.txt".format(driver_info))
            with open(save_file_train_s, 'a+') as af:
                for each in lines[train_s_sig_start_idx:train_s_sig_stop_idx]:
                    af.write("{}\n".format(each))

            save_file_test_ns = os.path.join(file_dir, "{}_ns_test.txt".format(driver_info))
            with open(save_file_test_ns, 'a+') as af:
                for each in lines[test_ns_sig_start_idx:test_ns_sig_stop_idx]:
                    af.write("{}\n".format(each))

            save_file_test_s = os.path.join(file_dir, "{}_s_test.txt".format(driver_info))
            with open(save_file_test_s, 'a+') as af:
                for each in lines[test_s_sig_start_idx:test_s_sig_stop_idx]:
                    af.write("{}\n".format(each))

            print("train test saved for {}".format(driver_info))

    def convert_rpeaks_to_img(files, savedir, drive):
        data = np.genfromtxt(files, dtype=None, encoding=None)
        #print(type(data), len(data), data)

        rpeaks = biosppy.signals.ecg.ecg(signal=data, sampling_rate=825 , show=False)['rpeaks']
        heartbeats = biosppy.signals.ecg.extract_heartbeats(signal=data, rpeaks=rpeaks, sampling_rate=825, before=0.2, after=0.2)[0]

        ecg_count = 1  
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for sig in heartbeats:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            plt.axis('off')
            plt.plot(sig)

            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(os.path.join(savedir, "{}_ecg_{}.jpg".format(drive, ecg_count)), bbox_inches=extent)
            print("{} ECG image {} saved".format(drive, ecg_count))
            plt.close()
            ecg_count += 1
    
    def generate_ecg_img():

        data_folder = "../Datasets/ECG/DRIVEDB/processed"
        train_folder = "../Datasets/ECG/DRIVEDB/processed/train"
        test_folder = "../Datasets/ECG/DRIVEDB/processed/test"

        for files in sorted(tqdm(glob.glob(data_folder + '/*'))):
            filename = os.path.basename(files)
            if "train" in filename:
                if "_ns_" in filename:
                    save_folder = os.path.join(train_folder, "non_stress")
                    driver = filename.split("_")[0]
                    convert_rpeaks_to_img(files, save_folder, driver)
                elif "_s_" in filename:
                    save_folder = os.path.join(train_folder, "stress")
                    driver = filename.split("_")[0]
                    convert_rpeaks_to_img(files, save_folder, driver)
            elif "test" in filename:
                if "_ns_" in filename:
                    save_folder = os.path.join(test_folder, "non_stress")
                    driver = filename.split("_")[0]
                    convert_rpeaks_to_img(files, save_folder, driver)
                elif "_s_" in filename:
                    save_folder = os.path.join(test_folder, "stress")
                    driver = filename.split("_")[0]
                    convert_rpeaks_to_img(files, save_folder, driver)


    #all_sig, meta = read_signals_and_meta()
    #display_indv_driver_info(meta)
    #save_indv_driver_ecg(all_sig, meta)
    #new_df = extracting_starting_time_df()
    #save_train_test_per_driver(new_df)
    generate_ecg_img()

#print(process_drivedb(drivedb_dataset_path))


def get_drivedb_img_label_list(data_path, mode):
    """
        Gets a list of lists, which contains [img_path, label] for each list

        Parameters:
        -----------
        data_path : string
            path to the data folder containing train/test folders
        mode : string
            defines whether "train" or "test" 

        Output:
        -------
        img_label_list : list
            list of lists, which contains [img_path, label] for each list
    """

    img_label_list = []

    folder_path = os.path.join(data_path, mode)

    for clas in sorted(tqdm(glob.glob(folder_path + '/*'))):
        class_name = os.path.basename(clas)
        label = -1
        if class_name == 'non_stress':
            label = 0
        elif class_name == 'stress':
            label = 1

        img_path = os.path.join(folder_path, class_name)
        
        for files in sorted(tqdm(glob.glob(img_path + '/*'))):
            print("file path is : {}".format(files))
            img_label_list.append([files, label])
    
    return img_label_list

#print(get_drivedb_img_label_list(data_path, 'test'))