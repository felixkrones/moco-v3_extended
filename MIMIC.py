import os
import os.path
import random
from typing import Dict
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


class MIMIC_Dataset(Dataset):
    """MIMIC dataset
    """

    def MIMIC_split_csv(self, data_fol: str):

        csv_path = os.path.join(data_fol, 'mimic-cxr-2.0.0-split.csv.gz')
        csv_arr = pd.read_csv(csv_path).to_numpy()

        img_fol = os.path.join(data_fol, 'files')

        data_paths = np.array([os.path.join(img_fol, 'p'+str(r[-2])[0:2], 'p'+str(r[-2]), 's'+str(r[-3]), r[0]+'.jpg') for r in csv_arr])

        paths = {
            'train': data_paths[np.where(csv_arr[:,-1]=='train')[0]],
            'validate' : data_paths[np.where(csv_arr[:,-1]=='validate')[0]],
            'test': data_paths[np.where(csv_arr[:,-1]=='test')[0]], 
            }

        return paths


    def __init__(self,
                 datapath,
                 split,
                 transform = None,
                 input_channels = 3,
                 input_size = 224,
                 seed = 42,
                 n_samples = None,
                 large_img = True,
                 ):
        super(MIMIC_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.MEAN_VAL = 0.5292
        self.STD_VAL = 0.2507
        self.datapath = datapath
        self.input_channels = input_channels
        self.input_size = input_size
        self.transform = transform

        if large_img:
            a1 = 'physionet.org'
            print("Using large images!")
        else:
            a1 = "images_224_224"
            print("Using small images!")
        a2 = 'files'
        a3 = 'mimic-cxr'
        a4 = '2.0.0'
        data_fol = os.path.join(datapath, a1, a2, a3+'-jpg', a4)
        paths = self.MIMIC_split_csv(data_fol)
        self.paths = paths[split]

        # Randomly sample n_samples images
        if n_samples is None:
            n_samples = len(self.paths)
        if n_samples < len(self.paths):
            print(f"Sampling the dataset to {n_samples} observations")
            self.paths = np.random.choice(self.paths, size=n_samples, replace=False)
        elif n_samples > len(self.paths):
            raise ValueError(f"n_samples must be less than or equal to the number of images in the dataset, which is {len(self.metadata)}")

        self.pathologies = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
        self.num_classes = len(self.pathologies)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img_path = self.paths[idx]
        if self.input_channels == 1:
            img = Image.open(img_path).convert("L")
        elif self.input_channels == 3:
            img = Image.open(img_path).convert("RGB")
        else:
            raise ValueError(f"input_channels must be 1 or 3, not {self.input_channels}")

        p = img_path.split('/')
        patient, study = p[-3], p[-2]

        #img = imread(img_path, as_gray=(not self.input_channels))
        img = self.transform(img)

        lab = ["None"]

        return img, lab