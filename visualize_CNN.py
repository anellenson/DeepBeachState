from __future__ import print_function, division

import os
from PIL import Image
import matplotlib.pyplot as pl
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from grad_cam import BackPropagation, GuidedBackPropagation, GradCAM
import pickle
import cv2
from collections import Counter
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
#
modelbasename = 'resnet512_five_aug_trainloss_'
runno = 0
ii = 5
modelname = modelbasename + str(runno)
#statenum_duck = {'Ref':'1331485200', 'LTT':'1340726400', 'TBR':'1339516800', 'RBB':'1393347600', 'LBT':'1385053200'}
statenum = {'Ref':'1436477406', 'LTT':'1408136407', 'TBR':'1411765207', 'RBB':'1322773228', 'LBT':'1445995807'}
trainsite = 'nbn'
testsite = 'nbn'

synthetic = False #if synthetic is false, then it will go to determine if it is plot one state
plot_one_state = False
beachstate = "Ref"
vcut = False
print('Visualizing for model {}'.format(modelname))
imgdir = {'nbn': '/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/',
            'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/'}

manuscript_plot_dir = '/home/aquilla/aellenso/Research/DeepBeach/resnet_manuscript/plots/'

def load_images(test_IDs, res_height, res_width):
    images = []
    raw_images = []

    for ID in test_IDs:
        image, raw_image = preprocess(ID, res_height, res_width)
        images.append(image)
        raw_images.append(raw_image)

    return images, raw_images

def preprocess(image_path, res_height, res_width):
    transform = transforms.Compose([transforms.Resize((res_height,res_width)), transforms.ToTensor()])
                                       #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert("RGB")
        raw_image = transform(image)

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
classes = ['Ref','LTT','TBR','RBB','LBT']
topk = 2 #only ask for the top choice

basedir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/'
torch.cuda.empty_cache()
modelpath = '{}/resnet_models/train_on_{}/{}.pth'.format(basedir, trainsite, modelname)
res_height = 512 #height
res_width = 512 #width
out_folder = 'model_output/train_on_{}/'.format(trainsite)
##load model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nb_classes = len(classes)

def generate_img_probs(out_folder, modelbasename, img_id, testsite, classes, numruns=10):
    labels = []
    for rr in range(numruns):
        with open(out_folder + modelbasename + '{}/cnn_preds.pickle'.format(rr), 'rb') as f:
            cnn_preds = pickle.load(f)

        img_fnames = cnn_preds['{}_testfiles'.format(testsite)]
        img_labels = cnn_preds['{}_CNN'.format(testsite)]
        img_idx = img_labels.index(img_id)

        labels.append(img_labels[img_idx])

    ensemble_probs = {}
    for ci, state in enumerate(classes):
        stateprobs = labels.count(ci)/numruns
        ensemble_probs[state] = stateprobs

    return ensemble_probs

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

if plot_one_state:
    vis_test_dir = vis_dir +'test_on_{}/'.format(testsite,beachstate)

else:
    vis_test_dir = vis_dir

#Remove all files that might already be there:

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

if not os.path.exists(vis_test_dir):
    os.mkdir(vis_test_dir)

if synthetic:
    allimgs = os.listdir(imgdir['duck'])
    test_IDs = [f for f in allimgs if 'synthetic' in f]
    test_IDs = test_IDs[1:]

else:
##load images here, preprocess all of them (don't do a dataset)
    with open('labels/{}_daytimex_testfiles.final.pickle'.format(testsite), 'rb') as f:
        test_IDs = pickle.load(f)

    if vcut:
        valfiles = [tt[:-3] + 'vcut.jpg' for tt in test_IDs]

    valfiles = [tt for tt in test_IDs if not any([sub in tt for sub in trans_names])]
    valfiles.sort()


    with open('labels/{}_labels_dict_five_aug.pickle'.format(testsite), 'rb') as f:
        labels_dict = pickle.load(f)

    #filter so you get each class
    test_labels= np.array([labels_dict[pid] for pid in valfiles])
    test_IDs = []
    class_pids = []
    if plot_one_state:
        beachstatenum = classes.index(beachstate)
        class_pids = np.where(test_labels == beachstatenum)[0]
        test_IDs += [valfiles[cc] for cc in class_pids[ii:ii+5]]

    elif not plot_one_state:
        for beachstate in classes:
            class_pids = [aa for aa in valfiles if statenum[beachstate] in aa]
            test_IDs += class_pids

test_IDs = [imgdir[testsite] + '/'+tt for tt in test_IDs]


images, raw_images = load_images(test_IDs, res_height, res_width)
images = torch.stack(images).to(device)

fig_cam, ax_cam = pl.subplots(5, topk + 1, figsize = [15,15])
fig_cam.subplots_adjust(0,0,0.9,1)

fig_gbp, ax_gbp = pl.subplots(5, topk + 1, figsize = [15,15])
fig_gbp.subplots_adjust(0,0,0.9,1)

fig_gbpcam, ax_gbpcam = pl.subplots(5, topk + 1, tight_layout = {'rect':[0,0, 1, 0.95]}, figsize = [10,15])
fig_gbpcam.subplots_adjust(0,0,0.9,1)
if testsite == 'nbn':
    Testsite = "Narrabeen"
if testsite == 'duck':
    Testsite = 'Duck'
pl.suptitle('Saliency Maps: Tested at {}'.format(Testsite), fontsize = 20)



all_gradcams = []
all_probs = {}

for j, (image, ID) in enumerate(zip(images, test_IDs)):

    image = image.unsqueeze(dim = 0)
    ID = ID.split('/')[-1]
    ID = ID.split('.')[0]
    if testsite == 'nbn':
        ID = ID.split('_')[1]

    for ax in [ax_cam, ax_gbp, ax_gbpcam]:
        ax[j,0].imshow(image.squeeze().cpu().numpy().transpose(1,2,0))
        ax[j,0].axis('off')
        if synthetic:
            ax[j,0].set_title('Mixed State {}'.format(ID), fontsize = 18)
        else:
            if plot_one_state:
                ax[j,0].set_title('{} {}'.format(beachstate, ID), fontsize = 18)

            elif not plot_one_state:
                ax[j,0].set_title('{}'.format(classes[j]), fontsize = 18)

    bp = BackPropagation(model=model_conv)
    probs, ids = bp.forward(image)#generate the top predictions

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

        #for ax in [ax_cam, ax_gbp, ax_gbpcam]:
        for ax in [ax_gbpcam]:
            ax[j,i+1].set_title(prediction, fontsize = 22)
            ax[j,i+1].imshow(image.detach().squeeze().cpu().numpy().transpose(1,2,0))
            ax[j,i+1].axis('off')

        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        #print("\t#{}: {} ({:.5f})".format(ii + j, classes[ids[0, i]], ensemble_probs[prediction]))
        print("\t#{}: {} ({:.5f})".format(ii + j, classes[ids[0, i]], probs[0,i]))

        # Guided Backpropagation
        gbp_gradient = gradient_im(gradients[0])
        ax_gbp[j,i+1].imshow(gbp_gradient)
        ax_gbp[j,i+1].set_title('{0} {1:.2f}'.format(prediction, probs[0,i]))
        ax_gbp[j,i+1].set_title('{0}'.format(prediction))

        #Grad-Cam
        grad_cam = regions[0,0].cpu().numpy()
        ax_cam[j,i+1].pcolor(np.flipud(grad_cam), alpha = 0.4, cmap = 'jet')
        ax_cam[j,i+1].set_title('{0} {1:.2f}'.format(prediction, probs[0,i]))
        ax_cam[j,i+1].set_title('{0}'.format(prediction))

        # Guided Grad-CAM
        guided_gradcam = gradient_im(torch.mul(regions, gradients)[0])
        guided_gradcam = guided_gradcam[:,:,0]
        #find the mode and zero it
        counts = Counter(list(guided_gradcam.flatten()))
        max_count = np.max(counts.values())
        mode = [k for k,v in counts.items() if v == max_count][0]
        #guided_gradcam[guided_gradcam>100] = 100
        guided_gradcam = guided_gradcam-mode
        #guided_gradcam[guided_gradcam<0] = 0
        #pl.imshow(guided_gradcam, cmap = 'hot', alpha = 0.4, vmin = 0, vmax = 100)

        ax_gbpcam[j, i+1].imshow(guided_gradcam, alpha = 0.5, cmap = 'hot', vmin = 0, vmax = 100)
        #ax_gbpcam[j,i+1].set_title('{0} {1:.2f}'.format(prediction, ensemble_probs[prediction]))
        ax_gbpcam[j,i+1].set_title('{0}'.format(prediction))

        if i == 0:
            all_gradcams.append(guided_gradcam)


all_gradcams = np.array(all_gradcams)
if plot_one_state:
    with open(vis_dir + '/{}_{}_imgprobs.pickle'.format(beachstate, ii), 'wb') as f:
        pickle.dump(all_gradcams, f)

    fig_cam.savefig(vis_test_dir + '/GradCam_{}_{}.png'.format(beachstate,ii), bbox_inches = 'tight')
    fig_gbp.savefig(vis_test_dir + '/Guided_Backprop_{}_{}.png'.format(beachstate,ii), bbox_inches = 'tight')
    fig_gbpcam.savefig(vis_test_dir + '/BackCAM_{}_{}.png'.format(beachstate,ii), bbox_inches = 'tight')
    if vcut:
        fig_gbpcam.savefig(vis_test_dir + '/BackCAM_{}_{}_vcut.png'.format(beachstate,ii), bbox_inches = 'tight')
    with open(vis_test_dir + '/BackCam_{}_{}.pickle'.format(beachstate, ii), 'wb') as f:
        pickle.dump(all_gradcams, f)


elif not plot_one_state:
    fig_cam.savefig(vis_test_dir + '/GradCam_{}_{}_{}.png'.format(testsite,beachstate,ii), bbox_inches = 'tight')
    fig_gbp.savefig(vis_test_dir + '/Guided_Backprop_{}_{}_{}.png'.format(testsite,beachstate,ii), bbox_inches = 'tight')
    fig_gbpcam.savefig(vis_test_dir + '/BackCAM_{}_{}_{}_withimg.png'.format(testsite,beachstate,ii), bbox_inches = 'tight')
    fig_gbpcam.savefig(manuscript_plot_dir + 'fig7_smap_{}.png'.format(testsite))
