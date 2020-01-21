import torch
from torch.utils import data as D
from PIL import Image
import os
from PIL import ImageFile


class ArgusTrainDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, basedirs, list_IDs, labels, gray = False, transform = None):
        """ Intialize the dataset
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.basedirs = basedirs #this is a list of basedirs, even if it's only one directory
        self.gray = gray #Switch if it's color or not


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index].encode('ascii')
        if 'tide' in ID:
            imgdir = [dir for dir in self.basedirs if 'Narrabeen' in dir][0]
            path = imgdir + ID

        if 'tide' not in ID:
            imgdir = [dir for dir in self.basedirs if 'north' in dir][0]
            path = imgdir + ID

        with open(path, 'rb') as f:
            X = Image.open(f)
            if self.gray: #if the input images are grayscle
                X.convert('RGB')
            if self.transform:
                X = self.transform(X)

        # Load data and get label
        y = self.labels[ID]


        return X, ID, y


class ArgusTestDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, basedirs, list_IDs, nbn = False, transform = None):
        """ Intialize the dataset
        """
        self.list_IDs = list_IDs
        self.transform = transform
        self.basedirs = basedirs
        #nbn is a true/false to toggle on narrabeen dataset
        self.nbn = nbn


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        ID = unicode(ID)

        if 'tide' in ID:
            imgdir = [dir for dir in self.basedirs if 'Narrabeen' in dir][0]
            path = imgdir + ID

        if 'tide' not in ID:
            imgdir = [dir for dir in self.basedirs if 'north' in dir][0]
            path = imgdir + ID


        with open(path, 'rb') as f:
            X = Image.open(f)
            if not self.nbn:
                X.convert('RGB') #nbn already in RGB
            if self.transform:
                X = self.transform(X)

        return X, ID

    def listIDS(self):
        return self.list_IDs


