from torch.utils.data import Dataset, DataLoader
import fnmatch
import pandas as pd
from PIL import Image, ImageFilter
from torchvision import transforms
import pickle
import os
import numpy as np
from PIL import Image
import random
import cv2
import imutils
import skimage
import numpy as np
import copy

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


class augmentFcns():

    def streaks(self, image):
        # This adds diagonal streaks by rotating the image 40 degrees, adding the streaks vertical, then unrotating it
        # The streaks are blurred parts of the image, and the returned image is slightly cropped.
        # Returns a PIL image

        #noise type options are gauss, s&p, speckle and poisson
        im_array = np.array(image)

        im_blur = image.filter(ImageFilter.GaussianBlur(radius = 20))
        im_blur_array = np.array(im_blur)

        im_rotate = imutils.rotate_bound(im_array, 40)
        im_blur_rotate = imutils.rotate_bound(im_blur_array, 40)

        #randomly generate a number of columns to choose to add streaks to:
        col_centers = np.random.randint(low = 50, high = 150, size = 2)
        cols = []

        for cc in col_centers:
            new_cols = list(np.arange(cc-3,cc+3))
            cols += new_cols

        cols = np.array(cols)

        im_rotate[:, cols] = im_blur_rotate[:, cols]


        im_straight = imutils.rotate_bound(im_rotate, -40)
        im_straight = im_straight[180:300, 50:400]

        im_straight = Image.fromarray(im_straight)

        return im_straight

    def find_spz(self, imgname):
        spz_path = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_spz/'
        img_dir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'
        path = img_dir + imgname


        img_files = os.listdir(spz_path)

        filename = imgname.split('.')
        days = filename[3].split('_')
        day = days[0]

        pattern = '*' + filename[2] + '.' + day + '*' + filename[-4] + '*'

        spzname = [ii for ii in img_files if fnmatch.fnmatch(ii, pattern)]

        if spzname == []:

            day_ = '{0:02d}'.format(int(day) + 1)
            pattern = '*' + filename[2] + '.' + day_ + '*' + filename[-4] + '*'
            spzname = [ii for ii in img_files if fnmatch.fnmatch(ii, pattern)]

        if spzname == []:

            _day = '{0:02d}'.format(int(day) - 1)
            pattern = '*' + filename[2] + '.' + _day + '*' + filename[-4] + '*'
            spzname = [ii for ii in img_files if fnmatch.fnmatch(ii, pattern)]

        try:
            img_path = spz_path + spzname[0]
            return img_path

        except:

            print('No SPZ file for {}'.format(imgname))



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

    def set_up_train_val(self, valfilename, trainfilename, num_train_imgs, num_val_imgs):

        '''
        This will set up partitions of train and validation sets as lists saved as pickles based off the entries in the labels dataframe
        It will ensure that the images in the dataframe and the images in the image directory are consistent

           num_train_imgs  =    number of images per class in training set
           num_val_imgs  =    number of images per class in validation set
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

            pids = list(self.labels_df[(self.labels_df['label'] == ci)].pid.values)


            trainfiles += pids[:num_train_imgs]
            valfiles += pids[num_train_imgs:num_train_imgs + num_val_imgs]

            print('Length of trainfiles is {} length of unique trainfiles is {}'.format(len(trainfiles), np.unique(len(trainfiles))))
            print('Length of valfiles is {} length of unique valfiles is {}'.format(len(valfiles), np.unique(len(valfiles))))

        random.shuffle(trainfiles)#shuffle the trainfiles later
        self.trainfiles = trainfiles
        self.valfiles = valfiles

        self.trainfilename = trainfilename
        self.valfilename = valfilename

        self.save_train_val(self.valfilename, self.trainfilename, self.valfiles, self.trainfiles)

    def load_train_val(self, valfilename, trainfilename):

        with open(valfilename, 'rb') as f:
            valfiles = pickle.load(f)

        self.valfiles = valfiles

        with open(trainfilename, 'rb') as f:
            trainfiles = pickle.load(f)

        self.trainfiles = trainfiles

        self.valfilename = valfilename
        self.trainfilename = trainfilename



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


    def augment_imgs(self, labels_dict_filename, augmentations):

        #Receives choice of augmentations

        #partition the files FIRST
        #Create the labels dictionary FIRST

        af = augmentFcns()
        trans_options = {'hflip': transforms.RandomHorizontalFlip(p = 1), 'vflip': transforms.RandomVerticalFlip(p = 1), 'rot':transforms.RandomRotation(15,fill=(0,)),
                         'erase':           transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.RandomErasing(p = 1, scale = (0.02, 0.08), ratio = (0.3, 3)),
                                            transforms.ToPILImage()]),
                         'affine':transforms.RandomAffine(0, translate = (.15, .20))}

        #loop through this twice (for valfiles and trainfiles)


        filenames = self.valfiles + self.trainfiles


        for augname in augmentations:
            #Each set of trainfiles are saved separately. The labels dictionary has all the augmentations ever made in it.

            self.trainfiles_aug = copy.copy(self.trainfiles)
            self.valfiles_aug = copy.copy(self.valfiles)

            for filenames in [self.trainfiles_aug, self.valfiles_aug]: #outer loop to keep consistency of trainfiles/valfiles

                for imgname in filenames:

                    if any([sub in imgname for sub in augmentations]):
                        continue

                    label = self.labels_dict[imgname]

                    path = self.img_dir + imgname

                    f = open(path, 'rb')
                    img = Image.open(f)
                    fi = None

                    if augname in trans_options.keys():
                        T = trans_options[augname]
                        img = T(img)
                        img.show()

                    elif augname == 'streaks':
                        img = af.streaks(img)

                    elif augname == 'spz':
                        img_path = af.find_spz(imgname)
                        if img_path is not None:
                            fi = open(img_path, 'rb')
                            img = Image.open(fi)

                        elif img_path is None:
                            continue

                    elif augname == 'gamma':
                        img = transforms.functional.adjust_gamma(img, gamma = 1.5)

                    elif augname == 'vcut':
                        img = transforms.functional.affine(img, 0, (0, -20), 1, 0)

                    elif augname == 'noise':
                        img = np.array(img)
                        img = skimage.util.random_noise(img, mode = 'gaussian', var = 0.001)
                        img = (img*255).astype(np.uint8)
                        img = Image.fromarray(img)

                    elif augname == 'noise.vcut.streaks':
                        img = transforms.functional.affine(img, 0, (0, -20), 1, 0)
                        img = af.streaks(img)
                        img = np.array(img)
                        img = skimage.util.random_noise(img, mode = 'gaussian', var = 0.001)
                        img = (img*255).astype(np.uint8)
                        img = Image.fromarray(img)

                    img = img.convert("L")

                #save out image file
                    filename = imgname[:-3] + augname + '.jpg'
                    img.save(self.img_dir + filename)

                    f.close()
                    if fi is not None:
                        fi.close()
                    # add to files list

                    filenames.append(filename)
                    entry = {filename:label}
                    self.labels_dict.update(entry)


            self.save_train_val(self.valfilename[:-13] + augname + '.pickle', self.trainfilename[:-13] + augname + '.pickle', self.valfiles_aug, self.trainfiles_aug)
            print('Finished producing/saving images from ' + augname + 'transformation')

        with open(labels_dict_filename, 'wb') as f:
            pickle.dump(self.labels_dict, f)


        #save out new labels dictionary


site = 'nbn'
img_dirs = {'duck':'/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn/', 'nbn':'/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}
labels_pickle = 'labels/{}_daytimex_labels_df.pickle'.format(site)
labels_df = pd.read_pickle(labels_pickle)

labels_dict_filename = 'labels/{}_labels_dict.pickle'.format(site)
img_folder = img_dirs[site]
valfilename = 'labels/{}_daytimex_valfiles.no_aug.pickle'.format(site)
trainfilename = 'labels/{}_daytimex_trainfiles.no_aug.pickle'.format(site)
num_train_imgs = 100
num_val_imgs = 15
augmentations = ['flips', 'rotate_darken', 'random_erasing']


F = File_setup(img_folder, labels_pickle)
#F.load_train_val(valfilename, trainfilename)
F.set_up_train_val(valfilename, trainfilename, num_train_imgs, num_val_imgs)
# be careful to not write over the validation files that you already have
F.create_labels_dict(labels_dict_filename)
F.augment_imgs(labels_dict_filename, augmentations)
