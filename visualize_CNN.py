from __future__ import print_function

import copy
import os.path as osp
import os
import click
from PIL import Image
import matplotlib.pyplot as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from grad_cam import BackPropagation, GuidedBackPropagation, GradCAM, Deconvnet
import pickle
import cv2
import shutil
import argparse


# parser = argparse.ArgumentParser(description = 'specify which model')
# parser.add_argument('-m', '--modelname')
# parser.add_argument('-state', '--beachstate')
# parser.add_argument('-ii', '--start_index', type = int)
# parser.add_argument('-testsite', '--testsite')
# parser.add_argument('-trainsite', '--trainsite')
# args = parser.parse_args()
# modelname = args.modelname
# beachstate = args.beachstate
# ii = args.start_index
# testsite = args.testsite
# trainsite = args.trainsite
modelname = 'resnet_aug_fulltrained'
ii = 0
beachstate = 'Ref'
trainsite = 'nbn'
testsite = 'nbn'
print('Visualizing for model {}'.format(modelname))

def load_images(test_IDs, res_height, res_width, mean, std):
    images = []
    raw_images = []

    for ID in test_IDs:
        image, raw_image = preprocess(ID, res_height, res_width, mean, std)
        images.append(image)
        raw_images.append(raw_image)

    return images, raw_images



def preprocess(image_path, res_height, res_width, mean, std):
    transform = transforms.Compose([transforms.Resize((res_height,res_width)), transforms.ToTensor()])
                                        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert("RGB")
        raw_image = transform(image)
        #image = transforms.Normalize([mean, mean, mean],[std, std, std])(raw_image)

    return raw_image, raw_image

def gradient_im(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0

    return gradient.astype(int)



def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.numpy().transpose(1,2,0)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


trans_names = ['hflip', 'vflip', 'rot', 'erase', 'gamma']
classes = ['Ref','LTT-B','TBR-CD','RBB-E','LBT-FG']
mean = 0.5199 # pull in from train_on_nbn/train_on_duck
std = 0.2319 #This is from the 'calc mean'
topk = 3 #only ask for the top choice
imgdir = {'duck':'/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/test/', 'nbn':'/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}
basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/'
torch.cuda.empty_cache()

modelpath= '{}/resnet_models/train_on_{}/{}.pth'.format(basedir, trainsite, modelname)
res_height = 512 #height
res_width = 512 #width

##load model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_classes = len(classes)

if 'resnet' in modelname:
    model_conv = models.resnet50()
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, nb_classes) # check is there really drop out


if 'inception' in modelname:
    from pretrainedmodels import inceptionresnetv2
    model_conv = inceptionresnetv2(pretrained=None, num_classes = nb_classes)

if 'mobilenet' in modelname:
    model_conv = models.mobilenet_v2()
    model_conv.classifier[1].out_features = nb_classes


model_conv.load_state_dict(torch.load(modelpath))
model_conv = model_conv.to(device)
model_conv.eval()

output_dir = 'model_output/train_on_{}/{}/'.format(trainsite,modelname)
vis_dir = output_dir + 'visualize/'
vis_test_dir = vis_dir +'test_on_{}/'.format(testsite,beachstate)

#Remove all files that might already be there:

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

if not os.path.exists(vis_test_dir):
    os.mkdir(vis_test_dir)

# beachstate_dir = full_test_dir  + beachstate + '/'
#
# if not os.path.exists(beachstate_dir):
#     os.mkdir(beachstate_dir)


##load images here, preprocess all of them (don't do a dataset)
with open('labels/{}_daytimex_valfiles.aug_imgs.pickle'.format(testsite), 'rb') as f:
    test_IDs = pickle.load(f)

valfiles = [tt for tt in test_IDs if not any([sub in tt for sub in trans_names])]


with open('labels/{}_labels_dict.pickle'.format(testsite), 'rb') as f:
    labels_dict = pickle.load(f)

#filter so you get each class
test_labels = np.array([labels_dict[pid] for pid in valfiles])
test_IDs = []
beachstatenum = classes.index(beachstate)
class_pids = np.where(test_labels == beachstatenum)[0]
test_IDs += [valfiles[cc] for cc in class_pids[ii:ii+5]]

# all_imgs = os.listdir('/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/test/')
# test_IDs = [tt for tt in all_imgs if 'synthetic' in tt]
# test_IDs = test_IDs[ii:ii+5]


test_IDs = [imgdir[testsite] + '/'+tt for tt in test_IDs]


images, raw_images = load_images(test_IDs, res_height, res_width, mean, std)
images = torch.stack(images).to(device)

fig_cam, ax_cam = pl.subplots(5, topk + 1, figsize = [15,15])
fig_cam.subplots_adjust(0,0,0.9,1)

fig_gbp, ax_gbp = pl.subplots(5, topk + 1, figsize = [15,15])
fig_gbp.subplots_adjust(0,0,0.9,1)

fig_gbpcam, ax_gbpcam = pl.subplots(5, topk + 1, figsize = [15,15])
fig_gbpcam.subplots_adjust(0,0,0.9,1)

for j, image in enumerate(images):
    image = image.unsqueeze(dim = 0)


    for ax in [ax_cam, ax_gbp, ax_gbpcam]:
        ax[j,0].imshow(image.squeeze().cpu().numpy().transpose(1,2,0))
        ax[j,0].axis('off')
        ax[j,0].set_title(beachstate)

    bp = BackPropagation(model=model_conv)
    probs, ids = bp.forward(image) #generate the top predictions

    gcam = GradCAM(model=model_conv)
    _ = gcam.forward(image)

    gbp = GuidedBackPropagation(model=model_conv)
    _ = gbp.forward(image)

    if 'resnet' in modelname:
        target_layer = 'layer4'

    if 'inception' in modelname:
        target_layer = 'conv2d_7b'

    if 'mobilenet' in modelname:
        target_layer = 'features' #can this just be features.18
    #target_layer = 'layer4' #which layer to pull from

    for i in range(topk):

        torch.cuda.empty_cache()

        prediction = classes[ids[0, i]]

        for ax in [ax_cam, ax_gbp, ax_gbpcam]:
            ax[j,i+1].set_title(prediction)
            #ax[j,i+1].imshow(image.detach().squeeze().cpu().numpy().transpose(1,2,0))
            ax[j,i+1].axis('off')



        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        print("\t#{}: {} ({:.5f})".format(ii + j, classes[ids[0, i]], probs[0, i]))

        # Guided Backpropagation
        gbp_gradient = gradient_im(gradients[0])
        ax_gbp[j,i+1].imshow(gbp_gradient)
        ax_gbp[j,i+1].set_title('{0} {1:.2f}'.format(prediction, probs[0,i]))

        #Grad-Cam
        grad_cam = regions[0,0].cpu().numpy()
        ax_cam[j,i+1].pcolor(np.flipud(grad_cam), alpha = 0.4, cmap = 'jet')
        ax_cam[j,i+1].set_title('{0} {1:.2f}'.format(prediction, probs[0,i]))

        # Guided Grad-CAM
        guided_gradcam = gradient_im(torch.mul(regions, gradients)[0])
        ax_gbpcam[j,i+1].imshow(guided_gradcam)
        ax_gbpcam[j,i+1].set_title('{0} {1:.2f}'.format(prediction, probs[0,i]))


for fig in [fig_cam, fig_gbp, fig_gbpcam]:
    fig.suptitle(modelname)

fig_cam.savefig(vis_test_dir + '/GradCam_{}_{}.png'.format(beachstate,ii), bbox_inches = 'tight')
fig_gbp.savefig(vis_test_dir + '/Guided_Backprop_{}_{}.png'.format(beachstate,ii), bbox_inches = 'tight')
fig_gbpcam.savefig(vis_test_dir + '/BackCAM_{}_{}.png'.format(beachstate,ii), bbox_inches = 'tight')

