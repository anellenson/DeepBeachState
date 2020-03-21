import datetime as dt
import pandas as pd
import os
from __builtin__ import any as b_any
import numpy as np
import matplotlib.pyplot as pl
import cv2
import scipy.io as sio
import pickle

labels_df = pd.read_pickle('../labels/nbn_labels_cleaned_165.pickle')
imgdir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_spz'

def transfer_pids_and_labels(labels_df):

    new_pids= os.listdir(imgdir)
    new_pids_dict ={}

    #This script will rename the files, keep the same label from the date:

    for pid in labels_df.pid:
        strnum = pid.split('.')[0]
        datenum = int(strnum.split('_')[1])
        datenum = int(datenum/1E5)

        if b_any(str(datenum) in p for p in new_pids):
            new_pid = [p for p in new_pids if str(datenum) in p][0]
            new_entry = {new_pid:labels_df[labels_df.pid == pid].label.values[0]}
            new_pids_dict.update(new_entry)

    return new_pids_dict

def count_labels_per_class(labels_df):
    for state in list(np.unique(labels_df.label.values)):
        print(state + ': {}'.format(len(labels_df[labels_df.label == state])))



# mat = {'pid':list(new_pids_dict.keys()), 'label':list(new_pids_dict.values())}
#
# sio.savemat('/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/nbn_labels.mat', mat)

########check how long they are:
# nbn_labels = sio.loadmat('/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/nbn_labels_relabel.mat')
#
# new_labels_df = pd.DataFrame({'pid':nbn_labels['pid'], 'label':nbn_labels['label']})

labels_df = pd.read_pickle('../labels/duck_daytimex_labels_df.pickle')


additional_labels = sio.loadmat('/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/HolmanEra_new_labelled_imgs_duck_0.mat')
pid = additional_labels['pid'][0]
pid = [pp[0] for pp in pid]

addl_labels_df = pd.DataFrame({'pid':pid, 'label':additional_labels['label'][1:]})
nbn_labels = pd.concat((addl_labels_df, labels_df))
nbn_labels = nbn_labels.drop_duplicates('pid')


nbn_labels['label'] = [ll.split()[0] for ll in nbn_labels.label]
#sort them
sorted_idx = np.argsort(nbn_labels.pid)
nbn_labels = nbn_labels.iloc[sorted_idx]

allfiles = os.listdir(imgdirs[trainsite])
newlabels = ['1394226027','1394658027','1396645228', '1400187607', '1400706007',
             '1427765407', '1431309606', '1433813407', '1436477406', '1458334806',
              '1462325407', '1462914006', '1475458207', '1525467606', '1529182806']
for ni, num in enumerate(newlabels):
    pid = [pp for pp in allfiles if num in pp][0]
    labels_df.loc[labels_df.index.max() + ni + 1] = ['Ref', pid]

labels_df = labels_df.drop_duplicates('pid')

labels_df.to_pickle('../labels/nbn_daytimex_labels_df.pickle')
# with open('missing_pids_labels.txt', 'wb') as f:
#     for pid in missing_pids:
#         f.write(pid + '\n')
#
# missing_labels = pd.read_csv('prep_files/missing_pids_labels.txt')
# new_pids = pd.DataFrame({'pid':list(new_pids_dict.keys()), 'label':list(new_pids_dict.values())})
#
# rename = {'LBT':'LBT-FG','RBB':'RBB-E','Ref':'Ref','TBR':'TBR-CD','LTT':'LTT-B'}
# for li,label in enumerate(missing_labels.label):
#     missing_labels.iloc[li].label = rename[label]
#
# daytimex_labels = pd.concat((new_pids,missing_labels))
#
# #How many entries of each class?
# daytimex_labels.to_pickle('labels/nbn_daytimex_labels.pickle')
#

# spz_files = os.listdir(imgdir)
# nbn_labels = pd.read_pickle('../nbn_daytimex_labels_df.pickle')
# extensions = ['erase', 'gamma', 'vflip', 'hflip', 'rot']

#load val files, save images to a directory, then write out csv with names
##########################################################3
#Write out validation set
import pickle
import random
import shutil

for trainsite in ['duck', 'nbn']:

    imgdirs = {'nbn': '/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/',
                'duck':'/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn/'}

    with open('labels/{}_daytimex_valfiles.no_aug.pickle'.format(trainsite), 'rb') as f:
        files = pickle.load(f)

    outdir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/valfiles/{}_images/'.format(trainsite)
    txtfname = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/valfiles/{}_images.txt'.format(trainsite)

    f = open(txtfname, 'wb')

    random.shuffle(files)
    for fi, file in enumerate(files):
        if trainsite == 'nbn':
            names = file.split('_')
            number = names[1].split('.')[0]

        if trainsite == 'duck':
            names = file.split('.')
            number = names[0]
        f.writelines('{0}.img.{1:02d}.{2}\n'.format(trainsite, fi, number))
        src = imgdirs[trainsite] + file
        dest = outdir + '{0}.img.{1:02d}.{2}.jpg'.format(trainsite, fi, number)
        shutil.copyfile(src,dest)

    f.close()
##############################################
#Clean out validation set and remake it

import matplotlib.pyplot as pl
from PIL import Image
import numpy as np

trainsite = 'nbn'

imgdirs = {'nbn': '/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/',
            'duck':'/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn/'}

labels_df = pd.read_pickle('labels/{}_daytimex_labels_df.pickle'.format(trainsite))
labels_df = labels_df.drop_duplicates('pid')

for pid in labels_df[labels_df.label == 'RBB-E'].pid:

    if pid in valfiles:

        image = Image.open(imgdirs[trainsite] + pid)
        fig, ax = pl.subplots(1,1)
        ax.set_title(pid)
        ax.imshow(image)



remove_imgs = ['1244667628', '1237669228', '1232571627', '1246136428', '1243371628', '1233954027', '1364763627', '1364158827']
pids = [pp for pp in valfiles if any([imgnum in pp for imgnum in remove_imgs])]
drop_inds = [labels_df[labels_df.pid == pp].index for pp in pids]
drop_inds = [107,295]
labels_df = labels_df.drop(index = drop_inds)

labels_df.drop(labels_df.index[drop_inds])
pids = [pp for pp in valfiles if '1252357227' in pp]
labels_df.loc[labels_df['pid'] == pids[0], 'label'] = 'LBT-FG'


####Remake valfiles
valfiles = []
for classname in ['Ref', 'LTT-B', 'TBR-CD', 'RBB-E', 'LBT-FG']:
    pids = labels_df[labels_df.label == classname].pid.values
    valfiles += list(pids[-15:])



#Check that duck valfiles are the later ones
trainsite = 'duck'
labels_df = pd.read_pickle('labels/{}_daytimex_labels_df.pickle'.format(trainsite))
labels_df = labels_df.drop_duplicates('pid')

pids = [int(ll.split('.')[0]) for ll in labels_df.pid]
sorted_idx = np.argsort(pids)


for classname in ['Ref', 'LTT-B', 'TBR-CD', 'RBB-E', 'LBT-FG']:
    pids = list(labels_df[labels_df.label == classname].pid)
    valfiles_class = [vv for vv in valfiles if vv in pids]
    print(classname + '\n ====================')
    print(pids[100])
    print(valfiles_class[-1])


def sort_pids(pids, trainsite):
    if trainsite == 'nbn':
        pids = [tt.split('.')[0] for tt in pids]
        pids = [int(tt.split('_')[1]) for tt in pids]

    if trainsite == 'duck':
        pids = [int(tt.split('.')[0]) for tt in pids]

    pids.sort()

    return pids




trainsite = 'nbn'
with open('labels/{}_daytimex_valfiles.final.pickle'.format(trainsite), 'wb') as f:
    pickle.dump(valfiles, f)


for state in ['Ref', 'LTT-B', 'TBR-CD', 'RBB-E', 'LBT-FG']:
#Check they're from different eras
    with open('labels/{}_daytimex_trainfiles.no_aug.pickle'.format(trainsite), 'rb') as f:
        trainfiles = pickle.load(f)


    with open('labels/{}_daytimex_valfiles.no_aug.pickle'.format(trainsite), 'rb') as f:
        valfiles = pickle.load(f)

    pids = list(labels_df[labels_df.label == state].pid)
    valfiles_class = [vv for vv in valfiles if vv in pids]
    trainfiles_class = [vv for vv in trainfiles if vv in pids]

    valfiles_class = sort_pids(valfiles_class, trainsite)
    trainfiles_class = sort_pids(trainfiles_class, trainsite)

    print(state + '\n ==================')
    print(trainfiles_class[-1]<valfiles_class[0])


