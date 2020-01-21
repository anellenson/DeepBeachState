import cv2
import matplotlib.pyplot as pl
from CAMplot import CAMplot
import pickle
import numpy as np
from PIL import Image
import math
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import interactive
interactive(True)
import pandas as pd

def display_convout_gridded(fxy):
    size = fxy.shape[-1]
    n_images = fxy.shape[0]
    n_cols = math.ceil(n_images**0.5)
    n_rows = math.ceil(n_images/n_cols)
    display_grid = np.ones((size * n_rows, size * n_cols))*np.nan
    cnt = 0
    for col in range(n_cols):
        for row in range(n_rows):
            while cnt < n_images:
                feature_set = fxy[cnt,:,:]
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = feature_set
                cnt += 1
    return display_grid


import pickle
f = open('model_output/nbn/train_on_duck_stretched/train_on_duck_stretched__run1_CNNprobs_testpids.pickle', 'rb')
model_output_stretched = pickle.load(f, encoding = 'latin1')
f.close()

f = open('model_output/nbn/train_on_duck/train_on_duck_run1_CNNprobs_testpids.pickle', 'rb')
model_output = pickle.load(f, encoding = 'latin1')
f.close()


class_names = ['Ref', 'LTT-B', 'TBR-CD', 'RBB-E', 'LBT-FG']
testpids = model_output['testpids']
CNNprobs = model_output['CNNprobs']

testpids_stretched = model_output_stretched['testpids']
CNNprobs_stretched = model_output_stretched['CNNprobs']

img_folder = r'C:\Users\z3530791\DeepBeach\images\Narrabeen_midtide_c5\orig_gray\\'
plotdir =  r'C:\Users\z3530791\DeepBeach\GitHub\python\ResNet\plots\nbn\confused_imgs_train_on_duck\\'
labels_df = pd.read_pickle("labels/nbn_labels_cleaned_165.pickle")
cnt = 0
for pi,pid in enumerate(testpids):
    label = class_names.index(labels_df[labels_df.pid == pid].label.values)
    probs = CNNprobs[pi]
    CNNpred = np.where(probs == np.max(probs))[0][0]

    pi_stretched = testpids.index(pid)
    probs_stretched = CNNprobs_stretched[pi_stretched]
    CNNpred_stretched = np.where(probs_stretched == np.max(probs_stretched))[0][0]

    if label != CNNpred:
        fig = pl.figure(1)
        fig, axes = pl.subplots(2,2)
        img = Image.open(img_folder + pid)
        img = img.resize((512,512))
        axes[0,0].imshow(img, cmap = 'gray')
        axes[0,0].axis('off')
        axes[0,0].set_title('Labelled {}, CNN {}'.format(class_names[label], class_names[CNNpred]))

        axes[1,0].scatter(range(len(class_names)), probs)
        axes[1,0].plot(range(len(class_names)), probs)
        axes[1,0].set_xticks(range(len(class_names)))
        axes[1,0].set_xticklabels(class_names)
        axes[1,0].set_ylim((0,1))

        img = img.resize((512, 256))
        axes[0,1].imshow(img, cmap = 'gray')
        axes[0,1].axis('off')
        axes[0,1].set_title('Labelled {}, CNN {}'.format(class_names[label], class_names[CNNpred_stretched]))

        axes[1,1].scatter(range(len(class_names)), probs_stretched)
        axes[1,1].plot(range(len(class_names)), probs_stretched)
        axes[1,1].set_xticks(range(len(class_names)))
        axes[1,1].set_xticklabels(class_names)
        axes[1,1].set_ylim((0, 1))


        pl.savefig(plotdir + '{}_confused_as_{}_{}_withprobs'.format(class_names[label], class_names[CNNpred_stretched], cnt))
        cnt += 1




#
# imgtype = 'raw'
#
# f = open('model_output/nbn/intermediate_test_{}.pickle'.format(imgtype), 'rb')
# model_output = pickle.load(f, encoding = 'latin1')
# f.close()
#
# modelname = 'nbn_images_h256_w128_ds150_{}_run0'.format(imgtype)
# class_names = ['Ref-Calm', 'LTT-B', 'LBT-CD', 'RBB-E', 'TBR-FG']
# imgdir = r'C:\Users\z3530791\DeepBeach\images\Narrabeen_midtide_c5\{}\\'.format(imgtype)
# allCAMs = model_output['CAM']
# imgs = model_output['pid']
# CNNprobs = model_output['CNNprobs']
#
#
# for ii in range(5):
#     mean_conv_out = []
#     for layer in ['layer4', 'layer3', 'layer2', 'layer1']:
#         mean_conv = np.mean(model_output[layer][ii].squeeze(), axis = 0)
#         mean_conv_out.append(mean_conv)
#
#     im = pl.imread(imgdir + imgs[ii])
#     CAM = allCAMs[ii]
#     CAM = CAM[0].squeeze()
#
#     im = cv2.resize(im, (512, 103))
#     CAM = cv2.resize(CAM, (512,103))
#     fig, ax = pl.subplots(2,1)
#     ax[0].imshow(im)
#     ax[0].axis('off')
#     ax[1].imshow(im)
#     ax[1].imshow(CAM, alpha = 0.4)
#     ax[1].axis('off')
#     pl.savefig('plots/nbn/{}/CAM_img{}.png'.format(modelname, ii))
#
#
#     w,h = im.shape[:2]
#
#     fig, ax = pl.subplots(4,1)
#     for mi, conv_map in enumerate(mean_conv_out):
#         conv_map = cv2.resize(conv_map, (h, w))
#         ax[mi].imshow(conv_map,cmap = 'jet')
#         ax[mi].axis('off')
#     pl.savefig('plots/nbn/{}/intlayers_img{}.png'.format(modelname, ii))
#
#
# fig, ax = pl.subplots(1,1)
# fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
# fig.set_size_inches(10,10)
# ax.pcolor(display_grid, cmap = 'jet')
# pl.grid(color = 'white')
# ax.axis('off')
# pl.show()
# fig.suptitle("fxy")
#
#
# for aa in range(len(CNNprobs)):
#     probs_CNN = CNNprobs[aa]
#     sorted_probs = np.argsort(CNNprobs[aa])
#     sorted_probs = sorted_probs[::-1]
#     CAMs = allCAMs[aa]
#     imgdir = r'C:\Users\z3530791\DeepBeach\images\Duck\rect_test\test\\'
#     img = Image.open(imgdir + imgs[aa])
#     img= img.resize((512,512))
#
#     pl.figure()
#     pl.imshow(img, cmap = 'gray')
#
#     fig, axes = pl.subplots(3,2)
#     fig.set_size_inches(3,5)
#     for ci,CAM in enumerate(CAMs):
#
#         ax = axes[ci,0]
#         ax.axis('off')
#         ax.imshow(img, cmap = 'gray')
#         #ax.imshow(CAM, alpha=0.2, cmap = 'jet')
#         ax.set_title(class_names[sorted_probs[ci]] + ' P = {0:.2f}'.format(probs_CNN[sorted_probs[ci]]))
#
#         ax = axes[ci,1]
#         ax.axis('off')
#         clr = ax.imshow(CAM, cmap = 'jet')
#         pl.colorbar(clr, ax = ax)
#
#     pl.subplots_adjust(wspace=None, hspace=None)
#     pl.savefig(r'C:\Users\z3530791\DeepBeach\GitHub\python\ResNet\plots\pure_images_h512_w512_more_labels_run3\CAMs\CNN_mixed_classes_{}'.format(aa))