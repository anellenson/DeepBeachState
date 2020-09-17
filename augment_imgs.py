import pandas as pd
from torchvision import transforms
import pickle
import os
import numpy as np
from PIL import Image
import random

class File_setup():

    def __init__(self, img_dir, labels_pickle, site):

        '''

        :param img_dir:             img directory where the images are stored
        :param labels_pickle:       labels dataframe where the labels are stored as 'pid' (picture ID) and 'label' (label as string)
        :param site:                choose Narrabeen = 'nbn' or Duck  = 'duck'
        '''
        self.labels_df = pd.read_pickle(labels_pickle)
        self.class_names = ['Ref','LTT','TBR','RBB','LBT']
        self.img_dir = img_dir
        self.site = site


    def set_up_train_test_val(self, trainfilename, percent_train, percent_val, percent_test, testfilename = None):

        '''
        This will set up partitions of train and validation sets as lists saved as pickles based off the entries in the labels dataframe.
        Labels_df is sorted chronologically.
        The testfiles are taken from the final set of images in the labels dataframe.
        It will ensure that the images in the dataframe and the images in the image directory are consistent.

        INPUTS:
           percent_train/val/test        percentage of data to be used as training/val/test
           valfilename                  name of pickle for validation files. The pickle will be a list of file names.
           trainfilename                picklename for trainfiles.
           testfilename                 optional - in the event that the test files are established and the training data is changed.

        OUTPUTS:
            trainfile, testfile and valfile list


        '''


        #Check to make sure the labels dataframe doesn't have names of any images that aren't in the image directory

        files = os.listdir(self.img_dir)
        missing_files = [ff for ff in self.labels_df.pid if ff not in files]
        missing_files_ind = [ii for ii in self.labels_df.index if self.labels_df[self.labels_df.index == ii].pid.values in missing_files]
        self.labels_df.drop(index = missing_files_ind) # remove missing files

        pids = labels_df.
        labels_df = labels_df.sort() # TODO - sort the labels dataframe chronologically

        print('Missing {} files from the labelled dataframe'.format(len(missing_files)))

        trainfiles = []
        valfiles = []

        if os.path.exists(testfilename):
            with open(testfilename, 'rb') as f:
                testfiles = pickle.load(f)

        else:
            testfiles = []

            for ci in self.class_names:
                pids = list(self.labels_df[(self.labels_df['label'] == ci)].pid.values)
                num_test = int(percent_test * len(pids))
                testfiles += pids[-num_test:]

        for ci in self.class_names:
            pids = list(self.labels_df[(self.labels_df['label'] == ci)].pid.values)
            #filter out any files that are in the validation set
            trainpids = [pp for pp in pids if pp not in testfiles]
            num_train_imgs = int(percent_train * len(pids))
            num_val_imgs = int(percent_val * len(pids))

            trainfiles += trainpids[:num_train_imgs]
            valfiles += trainpids[num_train_imgs:num_train_imgs+num_val_imgs]


            print('Length of trainfiles is {} length of unique trainfiles is {}'.format(len(trainfiles), np.unique(len(trainfiles))))
            print('Length of valfiles is {} length of unique valfiles is {}'.format(len(valfiles), np.unique(len(valfiles))))
            print('Length of testfiles is {} length of unique testfiles is {}'.format(len(testfiles), np.unique(len(testfiles))))

        self.trainfiles = trainfiles
        self.testfiles = testfiles
        self.valfiles = valfiles

        self.trainfilename = trainfilename
        self.valfilename = testfilename.split('.')[0][:-9] + 'valfiles.no_aug.pickle'

        self.save_train_val(self.valfilename, self.trainfilename, self.valfiles, self.trainfiles)

    def save_train_val(self, valfilename, trainfilename, valfiles, trainfiles):

        with open(valfilename, 'wb') as f:
            pickle.dump(valfiles, f)
        print('saved val files')

        with open(trainfilename, 'wb') as f:
            pickle.dump(trainfiles, f)

        print('saved train files')

    def create_labels_dict(self, labels_dict_filename):

        '''


        INPUTS:
                labels_dict_filename: name of the labels dictionary that is created from the labels dataframe
                labels_df

        OUTPUTS
                labels dictionary saved in labels/ folder


        '''

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

        '''
        This function will augment the training and validation dataset.
        INPUTS:
            labels_dict_filename:       the filename of the labels dictionary
            augmentations:              string of augmentation choices choices are 'hflip', 'vflip', 'rot', 'erase', 'translate' and 'gamma'

        If one wanted to add their own augmentation, a function could be added to this class. The function could be included below.

        OUTPUTS:
            Augments images based on augmentation choices
            trainfiles_aug:             list of names for trainfiles that are augmented
            valfiles_aug:               list of names for valfiles that are augmented

            These are saved as pickles in the labels folder
        '''

        trans_options = {'hflip': transforms.RandomHorizontalFlip(p = 1), 'vflip': transforms.RandomVerticalFlip(p = 1), 'rot':transforms.RandomRotation(15,fill=(0,)),
                         'erase':           transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.RandomErasing(p = 1, scale = (0.02, 0.08), ratio = (0.3, 3)),
                                            transforms.ToPILImage()]),
                         'translate':transforms.RandomAffine(0, translate = (.15, .20))}

        #loop through this twice (for valfiles and trainfiles)

        self.valfiles_aug = list(np.array(self.valfiles[:]).copy())
        self.trainfiles_aug = list(np.array(self.trainfiles[:]).copy())

        for fi,filenames in enumerate([self.valfiles, self.trainfiles]):

            for ti, name in enumerate(augmentations):

                for imgname in filenames:

                    if any([sub in imgname for sub in augmentations]):
                        continue

                    label = self.labels_dict[imgname]

                    path = self.img_dir + imgname

                    with open(path, 'rb') as f:
                        img = Image.open(f)

                        if name in trans_options.keys():
                            T = trans_options[name]
                            img = T(img)


                        elif 'flips' in name:
                            T = trans_options['hflip']
                            img = T(img)
                            T = trans_options['vflip']
                            img = T(img)


                        elif 'gamma' in name:
                            img = transforms.functional.adjust_gamma(img, gamma = 1.5)

                        img = img.convert("L")

                    #save out image file
                        filename = imgname[:-3] + name + '.jpg'
                        img.save(self.img_dir + filename)

                        if fi == 0:
                            self.valfiles_aug.append(filename)
                        if fi == 1:
                            self.trainfiles_aug.append(filename)

                    # add to files list

                    filenames.append(filename)
                    entry = {filename:label}
                    self.labels_dict.update(entry)


                print('Finished producing images from ' + name + 'transformation')

        # save out train and val files
        new_valname = self.valfilename.split('.')[0] + '.five_aug.pickle'
        new_trainname = self.trainfilename.split('.')[0] + '.five_aug.pickle'
        self.save_train_val(new_valname, new_trainname, self.valfiles_aug, self.trainfiles_aug)

        # save out labels dictionary
        with open(labels_dict_filename, 'wb') as f:
            pickle.dump(self.labels_dict, f)


        #save out new labels dictionary


site = 'duck'
img_dirs = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/', 'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}
labels_pickle = 'labels/{}_daytimex_labels_df.pickle'.format(site)
labels_df = pd.read_pickle(labels_pickle)

labels_dict_filename = 'labels/{}_labels_dict_five_aug.pickle'.format(site)
img_folder = img_dirs[site]
testfilename = 'labels/{}_daytimex_testfiles.final.pickle'.format(site)
trainfilename = 'labels/{}_daytimex_trainfiles.no_aug.pickle'.format(site)
percent_train = 0.8
percent_val = 0.2
augmentations = ['rot', 'flips', 'erase', 'trans', 'gamma']


F = File_setup(img_folder, labels_pickle, site)
F.create_labels_dict('labels/{}_labels_dict_no_aug.pickle'.format(site))
F.set_up_train_val(trainfilename, percent_train, percent_val, testfilename = testfilename)
F.augment_imgs(labels_dict_filename, augmentations)
