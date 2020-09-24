import torch
from torch.utils import data as D
from PIL import Image
import os


class ArgusTrainDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, basedir, list_IDs, labels, transform = None):
        """ Intialize the dataset
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.basedir = basedir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index].encode('ascii')
        path = self.basedir + ID

        with open(path, 'rb') as f:
            X = Image.open(f)
            if self.transform:
                X = self.transform(X)


        # Load data and get label
        y = self.labels[ID]


        return X, ID, y


class ArgusTestDS(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, basedir, list_IDs, transform = None):
        """ Intialize the dataset
        """

        self.list_IDs = list_IDs
        self.transform = transform
        for bb in basedir:
            if 'Narrabeen' in bb:
                self.basedir_nbn = bb
            if 'north' in bb:
                self.basedir_duck = bb



    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        ID = unicode(ID)

        if 'c5' in ID:
            path = self.basedir_nbn + ID

        if 'argus02' in ID:
            path = self.basedir_duck + ID


        with open(path, 'rb') as f:
            X = Image.open(f)
            X.convert('RGB')

        if self.transform:
            X = self.transform(X)

        return X, ID

    def listIDS(self):
        return self.list_IDs
