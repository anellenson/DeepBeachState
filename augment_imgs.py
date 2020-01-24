from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import preResnet as pre
import pickle
import os
import numpy as np
from PIL import Image
import random

class MyDataset(Dataset):
    def __init__(self, list_IDs, basedir, transform = None):
        self.list_IDs = list_IDs
        self.basedir = basedir
        self.transform = transform

    def __getitem__(self, index):
        ID = self.list_IDs[index].encode('ascii')

        path = self.basedir + ID

        with open(path, 'rb') as f:
            X = Image.open(f)
            X = self.transform(X)

        return X, ID

    def __len__(self):
        return len(self.list_IDs)


#This will calculate the mean and standard deviation

class File_setup():

    def __init__(self, img_dir, labels_pickle):

        self.labels_df = pd.read_pickle(labels_pickle)
        self.class_names = ['Ref','LTT-B','TBR-CD','RBB-E','LBT-FG']
        self.img_dir = img_dir



    def calc_mean_and_std(self, size = (256,256)):

        #This will provide the mean and variance of the given set of images.
        #size is a tuple of (h,w), default is 256x256 square image

        filenames = self.labels_df.pid.values
        trans = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

        dataset = MyDataset(filenames, img_dir, transform=trans)
        loader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=1,
            shuffle=False
        )


        mean = 0.
        std = 0.
        nb_samples = 0.
        for (data,_) in loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        print('The mean is {} and the std is {}'.format(mean, std))

        return mean, std


############set up validation dataset

    def set_up_train_val(self, valfilename, trainfilename, num_train_imgs):

        '''
        This will set up partitions of train and validation sets as lists saved as pickles based off the entries in the labels dataframe
        It will ensure that the images in the dataframe and the images in the image directory are consistent

           num_train_imgs  =    number of images per class in training set
           valfilename     =    name of pickle for validation files. The pickle will be a list of file names.
                                The number of images per class = (len(valfiles) + len(trainfiles)) - num_train_imgs


            trainfilename  =    picklename for trainfiles.
                                The number of images per class = num_train_images
        '''

        files = os.listdir(self.img_dir)
        missing_files = [ff for ff in self.labels_df.pid if ff not in files]
        self.labels_df.drop(index = missing_files) # remove missing files

        print('Missing {} files from the labelled dataframe'.format(len(missing_files)))

        trainfiles = []
        valfiles = []

        for ci in self.class_names: ####TO DO : make this so that it works with the labels dictionary, not necessarily a labels dataframe

            shuffled_inds = self.labels_df[(self.labels_df['label'] == ci)].index.values
            shuffled_inds = sorted(shuffled_inds, key=lambda k: random.random())
            total_len = len(shuffled_inds)

            if num_train_imgs > total_len:
                print('Number of desired training images exceeds the total number of images for ' + ci)


            for ss in shuffled_inds[:num_train_imgs]:
                trainfiles.append(self.labels_df.pid.iloc[ss])

            for ss in shuffled_inds[num_train_imgs:]:
                valfiles.append(self.labels_df.pid.iloc[ss])

        self.trainfiles = trainfiles
        self.valfiles = valfiles

        self.trainfilename = trainfilename
        self.valfilename = valfilename

        self.save_train_val(self.valfilename, self.trainfilename, self.valfiles, self.trainfiles)

    def save_train_val(self, valfilename, trainfilename, valfiles, trainfiles):

            with open(valfilename, 'wb') as f:
                pickle.dump(valfiles, f)
            print('saved val files')

            with open(trainfilename, 'wb') as f:
                pickle.dump(trainfiles, f)

            print('saved train files')

    def create_labels_dict(self, labels_dict_filename):

        labels_dict = {}

        for pid,label in zip(self.labels_df.pid, self.labels_df.label):
            if label in self.class_names: #Don't include the images you don't want
                classnum = self.class_names.index(label)
                entry = {pid:classnum}
                labels_dict.update(entry)

        with open(labels_dict_filename, 'wb') as f:
            pickle.dump(labels_dict, f)

        self.labels_dict = labels_dict

        return labels_dict


    def augment_imgs(self):


        #partition the files FIRST
        #Create the labels dictionary FIRST


        trans = [transforms.RandomHorizontalFlip(p = 1), transforms.RandomVerticalFlip(p = 1),
                                            transforms.RandomRotation(15,fill=(0,)),
                                            transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.RandomErasing(p = 1, scale = (0.02, 0.08), ratio = (0.3, 3)),
                                            transforms.ToPILImage()])]

        trans_names = ['hflip', 'vflip', 'rot', 'erase', 'gamma']

        #loop through this twice (for valfiles and trainfiles)

        self.valfiles_aug = self.valfiles[:]
        self.trainfiles_aug = self.trainfiles[:]

        for filenames in [self.valfiles_aug, self.trainfiles_aug]:


            for ti, name in enumerate(trans_names):

                for imgname in filenames:

                    if any([sub in imgname for sub in trans_names]):
                        continue

                    label = self.labels_dict[imgname]

                    path = self.img_dir + imgname

                    with open(path, 'rb') as f:
                        img = Image.open(f)

                        if ti < 4:
                            T = trans[ti]
                            img = T(img)

                        if ti == 4:
                            img = transforms.functional.adjust_gamma(img, gamma = 1.5)

                    #save out image file
                        filename = imgname[:-3] + name + '.jpg'
                        img.save(self.img_dir + filename)

                    # add to files list

                    filenames.append(filename)
                    entry = {filename:label}
                    self.labels_dict.update(entry)


                print('Finished producing images from ' + name + 'transformation')

        self.save_train_val(self.valfilename[:-6]+ 'aug_imgs.pickle', self.trainfilename[:-6]+ 'aug_imgs.pickle', self.valfiles_aug, self.trainfiles_aug)


labels_pickle = 'labels/duck_daytimex_labels_df.pickle'
labels_dict_filename = 'labels/duck_labels_dict.pickle'
img_folder = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/test/'
valfilename = 'labels/duck_daytimex_valfiles.pickle'
trainfilename = 'labels/duck_daytimex_trainfiles.pickle'
num_train_imgs = 75



F = File_setup(img_folder, labels_pickle)
F.set_up_train_val(valfilename, trainfilename, num_train_imgs)
F.create_labels_dict(labels_dict_filename)
F.augment_imgs()
