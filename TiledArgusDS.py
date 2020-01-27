import torch
from torch.utils import data as D
from PIL import Image
import os
import numpy as np
import cv2


class TiledArgusTrainDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, basedir, list_IDs, labels, resolution, transform = None):
        """ Intialize the dataset
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.basedir = basedir
        self.res = resolution
        all_imgs = os.listdir(self.basedir)
        #Sort the images according to the number
        imgtimestamps = []
        for img in all_imgs:
            timestamp = img.split('.')[0]
            imgtimestamps.append(int(timestamp))

        sorted_inds = [i[0] for i in sorted(enumerate(imgtimestamps), key=lambda x:x[1])]
        all_imgs = [all_imgs[ss] for ss in sorted_inds]
        self.all_imgs = all_imgs



    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index].encode('ascii')
        totalImg = torch.empty(3,self.res, self.res, dtype=torch.float)
        #Find where it is in the list, then select the four prior to that one
        img_idx = self.all_imgs.index(ID)
        prior_images = self.all_imgs[img_idx-3:img_idx+1]

        inputImages = []

        for pid in prior_images:
            path = self.basedir + pid
            with open(path, 'rb') as f:
                X = Image.open(f)
                X.convert('RGB')
                if self.transform:
                    X = self.transform(X)
                inputImages.append(X)

        totalImg[:,0:self.res/2, 0:self.res/2] = inputImages[0]
        totalImg[:,self.res/2:self.res, 0:self.res/2] = inputImages[1]
        totalImg[:,0:self.res/2, self.res/2:self.res] = inputImages[2]
        totalImg[:,self.res/2:self.res, self.res/2:self.res] = inputImages[3]


        # Load data and get label
        y = self.labels[ID]


        return totalImg, ID, y


class TiledArgusTestDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, basedir, list_IDs, resolution, transform = None):
        """ Intialize the dataset
        """
        self.list_IDs = list_IDs
        self.transform = transform
        self.basedir = basedir
        self.res  = resolution
        self.basedir = basedir
        all_imgs = os.listdir(self.basedir)
        #Sort the images according to the number
        imgtimestamps = []
        for img in all_imgs:
            timestamp = img.split('.')[0]
            imgtimestamps.append(int(timestamp))

        sorted_inds = [i[0] for i in sorted(enumerate(imgtimestamps), key=lambda x:x[1])]
        all_imgs = [all_imgs[ss] for ss in sorted_inds]
        self.all_imgs = all_imgs


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index].encode('ascii')
        totalImg = torch.empty(3,self.res, self.res, dtype=torch.float)
        #Find where it is in the list, then select the four prior to that one
        img_idx = self.all_imgs.index(ID)
        prior_images = self.all_imgs[img_idx-3:img_idx+1]

        inputImages = []

        for pid in prior_images:
            path = self.basedir + pid
            with open(path, 'rb') as f:
                X = Image.open(f)
                X.convert('RGB')
                if self.transform:
                    X = self.transform(X)
                inputImages.append(X)

        totalImg[:,0:self.res/2, 0:self.res/2] = inputImages[3]
        totalImg[:,self.res/2:self.res, 0:self.res/2] = inputImages[3]
        totalImg[:,0:self.res/2, self.res/2:self.res] = inputImages[3]
        totalImg[:,self.res/2:self.res, self.res/2:self.res] = inputImages[3]

        return totalImg, ID

    def listIDS(self):
        return self.list_IDs
