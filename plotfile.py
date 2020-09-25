from __future__ import division

import pickle
from PIL import Image
import matplotlib.pyplot as pl
import numpy as np

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
