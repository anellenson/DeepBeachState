from __future__ import print_function

import copy
import os.path as osp
import os
import click
from PIL import Image
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from grad_cam import BackPropagation, GuidedBackPropagation, GradCAM, Deconvnet
import pickle
import cv2
from shutil import copyfile
import argparse


# parser = argparse.ArgumentParser(description = 'specify which model')
# parser.add_argument('-m', '--modelname')
# args = parser.parse_args()
# modelname = args.modelname

def load_images(test_IDs, resolution, mean, std):
    images = []
    raw_images = []

    for ID in test_IDs:
        image, raw_image = preprocess(ID, resolution, mean, std)
        images.append(image)
        raw_images.append(raw_image)

    return images, raw_images



def preprocess(image_path, resolution, mean, std):
    transform = transforms.Compose([transforms.Resize((resolution,resolution)), transforms.ToTensor()])
                                        #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert("RGB")
        raw_image = transform(image)
        #image = transforms.Normalize([mean, mean, mean],[std, std, std])(raw_image)

    return raw_image, raw_image

def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, gradient.astype(int))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.numpy().transpose(1,2,0)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


trainsite = 'nbn'
testsite = 'nbn'
trans_names = ['hflip', 'vflip', 'rot', 'erase', 'gamma']
classes = ['Ref','LTT-B','TBR-CD','RBB-E','LBT-FG']
mean = 0.5199 # pull in from train_on_nbn/train_on_duck
std = 0.2319 #This is from the 'calc mean'
topk = 1 #only ask for the top choice
imgdir = {'duck':'/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn', 'nbn':'/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}
basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/'
modelname = 'resnet_aug_fulltrained'
torch.cuda.empty_cache()

modelpath= '{}/resnet_models/train_on_{}/{}.pth'.format(basedir, trainsite, modelname)
resolution = 512
##load model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_classes = len(classes)

if 'inception' not in modelname:
    model_conv = models.resnet50()
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, nb_classes) # check is there really drop out


if 'inception' in modelname:
    from pretrainedmodels import inceptionresnetv2
    model_conv = inceptionresnetv2(pretrained=None, num_classes = nb_classes)



model_conv.load_state_dict(torch.load(modelpath))
model_conv = model_conv.to(device)
model_conv.eval()

output_dir = 'model_output/train_on_{}/{}/'.format(trainsite,modelname)
vis_dir = output_dir + 'visualize/'
full_test_dir = vis_dir +'test_on_{}'.format(testsite)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

if not os.path.exists(full_test_dir):
    os.mkdir(full_test_dir)


##load images here, preprocess all of them (don't do a dataset)
with open('labels/{}_daytimex_valfiles.aug_imgs.pickle'.format(testsite), 'rb') as f:
    test_IDs = pickle.load(f)
valfiles = [tt for tt in test_IDs if not any([sub in tt for sub in trans_names])]


with open('labels/{}_labels_dict.pickle'.format(testsite), 'rb') as f:
    labels_dict = pickle.load(f)

#filter so you get two images per class
test_labels = np.array([labels_dict[pid] for pid in valfiles])
test_IDs = []
for i in range(5):
    class_pids = np.where(test_labels == i)[0]
    test_IDs += [valfiles[cc] for cc in class_pids[:2]]

test_IDs = [imgdir[testsite] + '/'+tt for tt in test_IDs]


images, raw_images = load_images(test_IDs, resolution, mean, std)
images = torch.stack(images).to(device)

for j, image in enumerate(images):
    image = image.unsqueeze(dim = 0)

    bp = BackPropagation(model=model_conv)
    probs, ids = bp.forward(image) #generate the top predictions

    gcam = GradCAM(model=model_conv)
    _ = gcam.forward(image)

    gbp = GuidedBackPropagation(model=model_conv)
    _ = gbp.forward(image)

    deconv = Deconvnet(model=model_conv)
    _ = deconv.forward(image)

    if 'inception' not in modelname:
        target_layer = 'layer4'

    if 'inception' in modelname:
        target_layer = 'conv2d_7b'


    #target_layer = 'layer4' #which layer to pull from

    for i in range(topk):

        torch.cuda.empty_cache()

        #Deconvolution
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        save_gradient(
            filename=osp.join(
                full_test_dir,
                "{}-{}-deconvnet-{}.png".format(j, modelname, classes[ids[0, i]]),
            ),
            gradient=gradients[0],
        )

        print('Deonv net printed')
        deconv.remove_hook()

        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()
        #
        # # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        print("\t#{}: {} ({:.5f})".format(j, classes[ids[0, i]], probs[0, i]))


        # Guided Backpropagation
        save_gradient(
            filename=osp.join(
                full_test_dir,
                "{}-{}-guided-{}.png".format(j, modelname, classes[ids[0, i]]),
            ),
            gradient=gradients[0],
        )

        # Grad-CAM
        save_gradcam(
            filename=osp.join(
                full_test_dir,
                "{}-{}-gradcam-{}-{}.png".format(
                    j, modelname, target_layer, classes[ids[0, i]]
                ),
            ),
            gcam=regions[0,0],
            raw_image=raw_images[0],
        )

        # Guided Grad-CAM
        save_gradient(
            filename=osp.join(
                full_test_dir,
                "{}-{}-guided_gradcam-{}-{}.png".format(
                    j, modelname, target_layer, classes[ids[0, i]]
                ),
            ),
            gradient=torch.mul(regions, gradients)[0],
        )

        img = cv2.imread(test_IDs[j])
        img = cv2.resize(img, (resolution, resolution))
        cv2.imwrite(osp.join(full_test_dir, "{}_original_image.jpg".format(j)), img)


