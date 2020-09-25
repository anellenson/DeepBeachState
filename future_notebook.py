from __future__ import division

import pickle
from PIL import Image
import matplotlib.pyplot as pl
from utils import augment
import numpy as np


##Augment images:
img_dirs = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/', 'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}

site = 'nbn'
labels_pickle = 'labels/{}_labels_dict.pickle'.format(site)
labels_dict_filename = 'labels/{}_daytimex_labels_dict_five_aug.pickle'.format(site) #non-augmented labels dictionary
img_folder = img_dirs[site]
testfilename = 'labels/{}_daytimex_testfiles.final.pickle'.format(site)
filename = 'labels/{}_daytimex_train_val_files.pickle'.format(site)
percent_train = 0.6
percent_val = 0.2
percent_test = 0.2
augmentations = ['rot', 'flips', 'erase', 'trans', 'gamma']

F1 = augment.File_setup(5, img_folder, labels_pickle, site)
F1.set_up_train_test_val(percent_train, percent_val, percent_test)
F1.augment_imgs(labels_dict_filename, augmentations)
F1.save_train_val(filename, testfilename)


site = 'duck'
img_dirs = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/', 'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}

labels_pickle = 'labels/{}_labels_dict.pickle'.format(site)
labels_dict_filename = 'labels/{}_daytimex_labels_dict_five_aug.pickle'.format(site) #non-augmented labels dictionary
img_folder = img_dirs[site]
testfilename = 'labels/{}_daytimex_testfiles.final.pickle'.format(site)
filename = 'labels/{}_daytimex_train_val_files.pickle'.format(site)

F2 = augment.File_setup(5, img_folder, labels_pickle, site)
F2.set_up_train_test_val(percent_train, percent_val, percent_test)
F2.augment_imgs(labels_dict_filename, augmentations)
F2.save_train_val(filename, testfilename)

merged_trainfilename = 'labels/nbn_duck_trainfiles.pickle'
merged_valfilename = 'labels/nbn_duck_valfiles.pickle'
merged_testfilename = 'labels/nbn_duck_testfiles.pickle'
labels_dict_filename = 'labels/nbn_duck_labels_dict_five_aug.pickle'
augment.merge_train_val_files(merged_trainfilename, merged_valfilename, merged_testfilename, labels_dict_filename, F1, F2)


#Skill Evaluation
#Global skill (F1), Per-State (Confusion Table)
#================================================



#Gradcam Plotting
#=================================
imgpath = 'images/daytimex_1546219806.Mon.Dec.31_12_30_06.AEST.2018.narrabn.c5.jpg'
imgname = imgpath.split('/')[1][:-4]
ggcampath = 'model_output/resnet512_five_aug_trainloss_3/ggcam_daytimex_1546219806.Mon.Dec.31_12_30_06.AEST.2018.narrabn.c5.pickle'
beachstates = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']


#Load Data
#===============================
with open(ggcampath, 'rb') as f:
    ggcam_dict = pickle.load(f)
img = Image.open(imgpath)
img = img.resize((512,512))

topk = len(ggcam_dict.keys()) - 2
probs = ggcam_dict.pop('probs')
ids = ggcam_dict.pop('ids')

fig, ax = pl.subplots(1, topk + 1, tight_layout = {'rect':[0,0, 1, 0.95]}, figsize = [10,15])
fig.subplots_adjust(0,0,0.9,1)
ax[0].imshow(img, cmap = 'gray')
ax[0].axis('off')

for j, (_, ggcam) in enumerate(ggcam_dict.items()):
    statenum = ids[j]
    prob = probs[j]

    beachstate_string = beachstates[statenum]
    ggcam = ggcam/ggcam.max()
    ax[j + 1].imshow(img, cmap = 'gray')
    ax[j + 1].imshow(img, cmap = 'gray')
    ax[j + 1].axis('off')
    ax[j + 1].imshow(ggcam, alpha = 0.5, cmap = 'hot')
    ax[j + 1].set_title('{0}, P={1:0.3f}'.format(beachstate_string, prob))

pl.show()
