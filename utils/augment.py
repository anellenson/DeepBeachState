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
        wd = os.getcwd()
        print(wd)
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

            testfilename                 In the event that the test files are established and the training data is changed, name the testfile

        OUTPUTS:
            trainfile, testfile and valfile list


        '''


        #Check to make sure the labels dictionary doesn't have names of any images that aren't in the image directory
        self.check_for_missing_files()

        #Sort the labels dictionary to choose images that are 'out of sample'
        labels_dict_ordered = self.sort_labels_dict()

        trainfiles = []
        valfiles = []
        if testfilename:
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


    def save_train_val(self, filename, testfilename, labels_dict_filename):

        '''

        :param trainfilename:   filename that the trainfiles will be saved to
        :param valfilename:     filename that the valfiles will be saved to
        :param testfilename:    filename that the testfiles will be saved to. If this already exists, nothing will happen.
        :return:
        '''

        files_dict = {'valfiles':self.valfiles, 'trainfiles':self.trainfiles}

        with open(filename, 'wb') as f:
            pickle.dump(files_dict, f)
        print('saved train/val files as {}'.format(filename))

        if not os.path.exists(testfilename):
            with open(testfilename, 'wb') as f:
                pickle.dump(self.testfiles, f)
            print('saved test files as {}'.format(testfilename))
        else:
            print('{} exists'.format(testfilename))

        with open(labels_dict_filename, 'wb') as f:
            pickle.dump(self.labels_dict, f)
        print('Saved new augmented dictionary at {}'.format(labels_dict_filename))


    def augment_imgs(self, augmentations):

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

            if ti == 0:
                print('Finished producing images from ' + name + 'transformation for validation dataset')
            if ti == 1:
                print('Finished producing images from ' + name + 'transformation for training dataset')
        # save out labels dictionary




