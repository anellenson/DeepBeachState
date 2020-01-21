import pandas as pd
import datetime as dt
import os
from shutil import copyfile
import numpy as np
import random
import pickle
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as pl
import cPickle


def loadLabels(matfilename, waveparams):
    #This will read in a matfile and create dictionaries of the image names and their label for the labels dictionary
    #This will also create a pandas dataframe of each image path and their corresponding class
    matfile = sio.loadmat(matfilename)
    matlabels = matfile['labels_final'][0][:]
    matfnames = matfile['labeled_files_final'][:]
    unique_ = np.where(matfnames == np.unique(matfnames))[1]
    matfnames = matfnames[0][unique_]
    matlabels = matlabels[unique_]
    labels = {} #to create the dictionary for the Argus Dataset
    label_list = [] #for entry in pandas data frame
    entry = {}
    waveconds = pd.read_pickle("/home/server/pi/homes/aellenso/Research/DeepBeach/data/image_bulkwaveparams.pickle")

    for idx in np.arange(len(matlabels)):
        fname = matfnames[idx][0]
        label = matlabels[idx]

        label = label - 1 #Remove offset from matlab indexing
        if label < 9:
            entry = {fname:label}
            labels.update(entry)
            label_list.append([fname,label])

    labels_df = pd.DataFrame(label_list, columns = ['file','label'])

    #Now add the wave parameters
    for ww in waveparams:
        labels_df[ww] = np.nan
        for ll in labels_df.file:
            fname = ll.encode('ASCII')
            try:
                labels_df.loc[labels_df.file == fname, ww] = waveconds[waveconds['pid'] == fname][ww].values[0]
            except IndexError:
                continue

    if waveparams != []:
        labels_df = labels_df[np.isfinite(labels_df[waveparams[0]])].reset_index()
    flabel = open('labels.pikl','wb')
    pickle.dump(labels,flabel,protocol = 2)
    flabel.close()
    labels_df.to_pickle("labels_df.pickle")

    return labels,labels_df

def createTrainValSets(labels_df,classes):
    #This will read in a pandas data frame, find the files belonging to each class, shuffle the files, and
    #store the first 80% in the training partition, and the final 20% in the validation partition
    trainfiles = []
    valfiles = []
    labels = {}

    for ci in classes:

        shuffled_inds = labels_df[(labels_df['label'] == ci)].index.values
        shuffled_inds = sorted(shuffled_inds, key=lambda k: random.random())
        total_len = len(shuffled_inds)
        train_len = total_len - 15

        for ss in shuffled_inds[0:train_len]:
            trainfiles.append(labels_df.pid.iloc[ss])
            entry = {labels_df.pid.iloc[ss]:classes.index(labels_df.label.iloc[ss])}
            labels.update(entry)


        for ss in shuffled_inds[train_len:]:
            valfiles.append(labels_df.pid.iloc[ss])
            entry = {labels_df.pid.iloc[ss]:classes.index(labels_df.label.iloc[ss])}
            labels.update(entry)

    partition = {'train':trainfiles,'val':valfiles}
    return partition, labels

def createLabelsDict(labels_df, class_names):
    pids = []
    labels = []

    for ci in class_names:
        pidset = [pp for pp in labels_df[(labels_df['label'] == ci)].pid]

        pids = pids + pidset

    #Shuffle the files
    random.shuffle(pids)

    #Create the labels dictionary
    labels_dict = {}

    for pid in pids:
        classnum = class_names.index(labels_df[labels_df.pid == pid]['label'].values[0])
        labels.append(classnum)
        entry = {pid:classnum}
        labels_dict.update(entry)

    return pids, labels, labels_dict

def equalize_classes(class_names, valfiles, labels_df, num):
    trainfiles = []
    for state in class_names:
        inds_class = np.where(labels_df.label == state)[0]
        train_files_for_class = [labels_df.iloc[ii].pid for ii in inds_class if labels_df.iloc[ii].pid not in valfiles]
        train_files_for_class = train_files_for_class[:num]
        trainfiles = trainfiles + train_files_for_class

    return trainfiles

def calcMean(basedir,matfilename, transform, res1, res2):
    matfile = sio.loadmat(matfilename)
    matfnames = matfile['labeled_files_final'][0][:]

    batch_size = 10
    total_batches = np.round(len(matfnames))/batch_size
    all_mean = []
    all_std = []
    for tt in range(total_batches):
        start_ind = tt*batch_size
        all_imgs = np.empty((3,res1,res2))
        for idx in range(start_ind, start_ind + batch_size):
            ID = matfnames[idx][0]
            ID = ID.encode('ascii')
            path = basedir + ID

            with open(path, 'rb') as f:
                X = Image.open(f)
                X.convert('RGB')

            X = transform(X)
            all_imgs = np.concatenate((all_imgs, X.data.numpy()))
            mean_batch = np.mean(all_imgs[1:])
            std_batch = np.std(all_imgs[1:])
            all_mean.append(mean_batch)
            all_std.append(std_batch)
    mean = np.mean(all_mean)
    std = np.mean(all_std)

    return mean, std

def parsefname(filename):
    elements = filename.split('.')
    stringdate = elements[2] + '.' +  elements[3][0:2] + '.' + elements[5]
    date = dt.datetime.strptime(stringdate, '%b.%d.%Y')
    return date

def imshow(inp, mean, std, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([mean, mean, mean])
    std = np.array([std, std, std])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    pl.imshow(inp)
    if title is not None:
        plt.title(title)
    pl.pause(0.001)  # pause a bit so that plots are updated


def renamePartition(partition):
    for part in ['val','train']:
        for pi,pp in enumerate(partition[part]):
            n_ind = pp.index('oblique')
            new_name = pp[:n_ind-1] + '.rect.jpg'
            partition[part][pi] = new_name
    return partition

def countClasses(labels):
    #Pass a dictionary of labels to find the number of images per class
    num_imgs_perclass = []
    unique_classes = np.unique(list(labels.values()))
    label_vals = np.array([int(x) for x in list(labels.values())])

    for ci in unique_classes:
        num = len(np.where(label_vals == ci)[0])
        num_imgs_perclass.append(num)

    return num_imgs_perclass

def load_KFold_partition(pids):
    import scipy.io as sio
    kfold_inds = sio.loadmat('kfold_traintestsplit.mat')
    kfold_train_index = kfold_inds['traininds'][0]
    kfold_test_index = kfold_inds['testinds'][0]

    partition = {}
    split = 0
    for train_index, test_index in zip(kfold_train_index, kfold_test_index):
        trainfiles =[pids[pp] for pp in train_index[0]]
        testfiles = [pids[pp] for pp in test_index[0]]
        partition['train'] = trainfiles
        partition['val'] = testfiles

    return partition

def add_segmented_images(add_segment_states, partition, labels_dict, state_list):
    segmented_labels = pd.read_pickle('labels/segmented_labels.pickle')
    additional_labels = []
    for filename, state in segmented_labels.items():
        if state in add_segment_states:
            partition['train'].append(filename)
            state_num = state_list.index(state)
            new_entry = {filename: state_num}
            labels_dict.update(new_entry)

    return partition, labels_dict
