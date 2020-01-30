import datetime as dt
import pandas as pd
import os
from __builtin__ import any as b_any
import numpy as np
import matplotlib.pyplot as pl
import cv2
import scipy.io as sio

labels_df = pd.read_pickle('../labels/nbn_labels_cleaned_165.pickle')
imgdir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full'

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
nbn_labels = sio.loadmat('/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/nbn_labels_relabel.mat')

new_labels_df = pd.DataFrame({'pid':nbn_labels['pid'], 'label':nbn_labels['label']})

additional_labels = sio.loadmat('/home/server/pi/homes/aellenso/Research/DeepBeach/matlab/new_labelled_imgs_nbn')
pid = additional_labels['pid'][0]
pid = [pp[0] for pp in pid]

addl_labels_df = pd.DataFrame({'pid':pid, 'label':additional_labels['label']})
nbn_labels = pd.concat((addl_labels_df, new_labels_df))
nbn_labels = nbn_labels.drop_duplicates()

nbn_labels.to_pickle('../labels/nbn_daytimex_labels_df.pickle')
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
