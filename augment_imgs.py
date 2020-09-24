import pandas as pd
from torchvision import transforms
import pickle
import os
import numpy as np
from PIL import Image
import random
from collections import OrderedDict

class File_setup():

    def __init__(self,num_classes, img_dir, labels_pickle, site):

        '''

        :param img_dir:             img directory where the images are stored
        :param labels_pickle:       labels dictionary where the labels are stored as 'pid' (picture ID) and 'label' (label as string)
        :param site:                choose Narrabeen = 'nbn' or Duck  = 'duck'
        '''
        with open(labels_pickle, 'rb') as f:
             pickle.load(f)
        self.labels_dict = pd.read_pickle(labels_pickle)
        self.img_dir = img_dir
        self.site = site
        self.num_classes = num_classes

    def check_for_missing_files(self):
        '''
        This is a check to make sure that the labels dictionary and the images folder have the same images, otherwise the
        augmentations won't work.

        :return: labels_dictionary within self that
        '''

        files = os.listdir(self.img_dir)
        missing_files = [ff for ff in self.labels_dict.keys() if ff not in files]
        print('The image directory is missing {} files from the labelled dictionary'.format(len(missing_files)))

        for file in missing_files:
            self.labels_df.drop(file)
        # missing_files_ind = [ii for ii in self.labels_df.index if self.labels_df[self.labels_df.index == ii].pid.values in missing_files]
        # self.labels_df.drop(index = missing_files_ind) # remove missing files

    def find_datenum(self, pid):

        if self.site == 'duck':
            datenum =int(pid.split('.')[0])
        if self.site == 'nbn':
            datenum = int(pid.split('.')[0].split('_')[1])

        return datenum


    def sort_labels_dict(self):
        #Retrieve all pids, sort them, and return
        pids = self.labels_dict.keys()
        pids.sort(key=self.find_datenum)

        labels_dict_ordered = OrderedDict()
        for pp in pids:
            labels_dict_ordered.update({pp:self.labels_dict[pp]})

        return labels_dict_ordered


    def set_up_train_test_val(self, percent_train, percent_val, percent_test, testfilename = None):

        '''
        This will set up partitions of train and validation sets as lists saved as pickles based off the entries in the labels dataframe.
        Labels_df is sorted chronologically.
        The testfiles are taken from the final set of images in the labels dataframe.
        It will ensure that the images in the dataframe and the images in the image directory are consistent.

        INPUTS:
           percent_train/val/test        percentage of data to be used as training/val/test

           testfilename                 optional - in the event that the test files are established and the training data is changed.

        OUTPUTS:
            trainfile, testfile and valfile list


        '''


        #Check to make sure the labels dictionary doesn't have names of any images that aren't in the image directory
        self.check_for_missing_files()

        #Sort the labels dictionary to choose images that are 'out of sample'
        labels_dict_ordered = self.sort_labels_dict()

        trainfiles = []
        valfiles = []

        if os.path.exists(testfilename):
            with open(testfilename, 'rb') as f:
                testfiles = pickle.load(f)

        else:
            testfiles = []

            for ci in range(self.num_classes):
                pids = [pid for pid,label in labels_dict_ordered.items() if label == ci]
                num_test = int(percent_test * len(pids))
                testfiles += pids[-num_test:]


        for ci in range(self.num_classes):
            pids =  [pid for pid,label in labels_dict_ordered.items() if label == ci]

            #filter out any files that are in the validation set
            trainpids = [pp for pp in pids if pp not in testfiles]
            num_train_imgs = int(percent_train * len(pids))
            num_val_imgs = int(percent_val * len(pids))

            trainfiles += trainpids[:num_train_imgs]
            valfiles += trainpids[num_train_imgs:num_train_imgs+num_val_imgs]

        self.trainfiles_base = trainfiles
        self.valfiles_base = valfiles
        self.testfiles = testfiles


    def save_train_val(self, trainfilename, valfilename, testfilename):

        '''

        :param trainfilename:   filename that the trainfiles will be saved to
        :param valfilename:     filename that the valfiles will be saved to
        :param testfilename:    filename that the testfiles will be saved to. If this already exists, nothing will happen.
        :return:
        '''


        with open(valfilename, 'wb') as f:
            pickle.dump(self.valfiles, f)
        print('saved val files as {}'.format(valfilename))

        with open(trainfilename, 'wb') as f:
            pickle.dump(self.trainfiles, f)

        print('saved train files as {}'.format(trainfilename))

        if not os.path.exists(testfilename):
            with open(testfilename, 'wb') as f:
                pickle.dump(self.trainfiles, f)
            print('saved test files as {}'.format(testfilename))
        else:
            print('{} exists'.format(testfilename))

    def augment_imgs(self, labels_dict_filename, augmentations):

        '''
        This function will augment the training and validation dataset.
        INPUTS:
            labels_dict_filename:       the filename of the labels dictionary to be fed into CNN
            augmentations:              string of augmentation choices choices are 'hflip', 'vflip', 'rot', 'erase', 'translate' and 'gamma'

        If one wanted to add their own augmentation, a function could be added to this class. The function could be included below.

        OUTPUTS:
            Augments images based on augmentation choices
            trainfiles_aug:             list of names for trainfiles that are augmented
            valfiles_aug:               list of names for valfiles that are augmented
            labels_dict_

            These are saved as pickles in the labels folder
        '''

        trans_options = {'hflip': transforms.RandomHorizontalFlip(p = 1), 'vflip': transforms.RandomVerticalFlip(p = 1), 'rot':transforms.RandomRotation(15,fill=(0,)),
                         'erase':           transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.RandomErasing(p = 1, scale = (0.02, 0.08), ratio = (0.3, 3)),
                                            transforms.ToPILImage()]),
                         'translate':transforms.RandomAffine(0, translate = (.15, .20))}

        #loop through this twice (for valfiles and trainfiles)

        self.valfiles = list(np.array(self.valfiles_base[:]).copy())
        self.trainfiles = list(np.array(self.trainfiles_base[:]).copy())

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
                            self.valfiles.append(filename)
                        if fi == 1:
                            self.trainfiles.append(filename)

                    # add to files list

                    filenames.append(filename)
                    entry = {filename:label}
                    self.labels_dict.update(entry)


                print('Finished producing images from ' + name + 'transformation')

        # save out labels dictionary
        with open(labels_dict_filename, 'wb') as f:
            pickle.dump(self.labels_dict, f)

def merge_train_val_files(trainfilename, valfilename, testfilename, labels_dict_filename, *f_objects):
    '''
    This function merges the trainfiles from different sites

    INPUTS:

        trainfilename:      merged trainfilename
        valfilename:        merged valfilename
        testfilename:       merged testfilename
        labels_filename:    merged labels dictionary filename
        *args:              File_setup objects with the trainfile set up names

    OUTPUT

        merged train, validation, and testfiles saved in locations indicated
    '''
    trainfiles = []
    valfiles = []
    testfiles = []
    labels_dict = {}

    for f_obj in f_objects:
        trainfiles += f_obj.trainfiles
        valfiles += f_obj.valfiles
        testfiles += f_obj.testfiles
        labels_dict.update(f_obj.labels_dict)


    with open(trainfilename, 'wb') as f:
        pickle.dump(trainfiles, f)

    with open(valfilename, 'wb') as f:
        pickle.dump(valfiles, f)

    with open(testfilename, 'wb') as f:
        pickle.dump(trainfiles, f)

img_dirs = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/', 'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}

site = 'nbn'
labels_pickle = 'labels/{}_labels_dict.pickle'.format(site)
labels_dict_filename = 'labels/{}_daytimex_labels_dict_five_aug.pickle'.format(site) #non-augmented labels dictionary
img_folder = img_dirs[site]
testfilename = 'labels/{}_daytimex_testfiles.final.pickle'.format(site)
trainfilename = 'labels/{}_daytimex_trainfiles.pickle'.format(site)
valfilename = 'labels/{}_daytimex_valfiles.pickle'.format(site)
percent_train = 0.6
percent_val = 0.2
percent_test = 0.2
augmentations = ['rot', 'flips', 'erase', 'trans', 'gamma']


F1 = File_setup(5, img_folder, labels_pickle, site)
F1.set_up_train_test_val(percent_train, percent_val, percent_test, testfilename = testfilename)
F1.augment_imgs(labels_dict_filename, augmentations)
F1.save_train_val(trainfilename, valfilename, testfilename)


site = 'duck'
img_dirs = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/', 'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}

labels_pickle = 'labels/{}_labels_dict.pickle'.format(site)
labels_dict_filename = 'labels/{}_daytimex_labels_dict_five_aug.pickle'.format(site) #non-augmented labels dictionary
img_folder = img_dirs[site]
testfilename = 'labels/{}_daytimex_testfiles.final.pickle'.format(site)
trainfilename = 'labels/{}_daytimex_trainfiles.pickle'.format(site)
valfilename = 'labels/{}_daytimex_trainfiles.pickle'.format(site)

F2 = File_setup(5, img_folder, labels_pickle, site)
F2.set_up_train_test_val(percent_train, percent_val, percent_test, testfilename = testfilename)
F2.augment_imgs(labels_dict_filename, augmentations)
F2.save_train_val(trainfilename, valfilename, testfilename)

merged_trainfilename = 'labels/nbn_duck_trainfiles.pickle'
merged_valfilename = 'labels/nbn_duck_valfiles.pickle'
merged_testfilename = 'labels/nbn_duck_testfiles.pickle'
labels_dict_filename = 'labels/nbn_duck_labels_dict_five_aug.pickle'
merge_train_val_files(merged_trainfilename, merged_valfilename, merged_testfilename, labels_dict_filename, F1, F2)

