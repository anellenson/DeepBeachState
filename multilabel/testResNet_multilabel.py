from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import ArgusDS
import pickle
import pandas as pd
import preResnet as pre
import os
import cv2
import postResnet as post
import TiledArgusDS

plt.ion()
#Resolution for tiled ds, res1/res2 for nontiled
resolution = 512
mean = 0.48
std = 0.29 #This is from the 'calc mean'
type = 'oblique'    
class_names = ['B','C','D','E','F','G','Calm','NoVis']
no_class = [70, 172, 91, 57, 107, 105, 134, 60]
weights = [1/np.cbrt(no) for no in no_class]
class_weights = torch.FloatTensor(weights).cuda()
plotfolder = '/home/server/pi/homes/aellenso/Research/DeepBeach/plots/resnet/prediction_prob/'



basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/'
test_transform = transforms.Compose([transforms.Resize((512,512)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize([mean, mean, mean],[std, std, std]),
                                ])

def calc_prob(conf_dt):
    total_perclass = np.sum(conf_dt, axis = 1)
    prob_perclass = conf_dt.values/total_perclass.values
    prob_df = pd.DataFrame(data = prob_perclass, columns = conf_dt.columns, index = conf_dt.index)
    return prob_df


def imshow(inp, mean, std, ax = None, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([mean, mean, mean])
    std = np.array([std, std, std])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def test_model(model, dataloader):

    model.eval()   # Set model to evaluate mode

    totalpreds = []
    totalprobs = []

    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            totalpreds.append(preds)

            if i % 100 == 0:
                print("On iteration {}".format(i))

    totalprobs_array = np.empty((len(totalprobs), len(totalprobs[0][0])))
    for ti,probs in enumerate(totalprobs):
        totalprobs_array[ti,:] = probs.cpu().numpy()

    return totalprobs_array

def visualize_model(model, plotfolder, class_names, label, num_images=1):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, inputs in enumerate(test_dl):
            inputs = inputs.to(device)
            fig, ax = plt.subplots(2,1)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax[0].axis('off')
                ax[0].set_title('predicted: {0:s}, class: {1:s}'.format(class_names[preds[j]], label))
                imshow(inputs.cpu().data[j], mean, std, ax = ax[0])
                ax[1].scatter(np.arange(len(probs[0])), probs.cpu())
                ax[1].plot(np.arange(len(probs[0])),probs[0].cpu().numpy())
                ax[1].set_xlim((0,8))
                ax[1].set_xticklabels(class_names)
                plt.savefig(plotfolder+'/img{0:2d}'.format(i), dpi = 400)


        model.train(mode=was_training)

def visualize_model_withCAM(model, plotfolder, class_names, label, test_dl, num_images=1):
    was_training = model.training
    model.eval()
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get('layer4').register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (resolution, resolution)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for bb in np.arange(bz):
            for idx in class_idx:
                cam = weight_softmax[idx].dot(feature_conv[bb].reshape((nc, h*w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    images_so_far = 0
    with torch.no_grad():
        for i, inputs in enumerate(test_dl):
            inputs = inputs.to(device)
            fig, ax = plt.subplots(2,1)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs)
            _, preds = torch.max(outputs, 1)
            CAMs = returnCAM(features_blobs[0], weight_softmax, preds.cpu().numpy())

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax[0].axis('off')
                ax[0].set_title('predicted: {0:s}, class: {1:s}'.format(class_names[preds[j]], label))
                imshow(inputs.cpu().data[j], mean, std, ax = ax[0])
                img = ax[0].imshow(CAMs[j], alpha = 0.4)
                plt.colorbar(img)

                ax[1].scatter(np.arange(len(probs[0])), probs.cpu())
                ax[1].plot(np.arange(len(probs[0])),probs[0].cpu().numpy())
                ax[1].set_xlim((0,8))
                ax[1].set_xticklabels(class_names)
                plt.savefig(plotfolder+'/img{0:2d}'.format(i), dpi = 400)
            plt.close()

        plt.close()
        model.train(mode=was_training)

#Create a dataloader from a list of filenames, or test_IDs
basedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/oblique/test/'
labels_df = pd.read_pickle('/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/labels/strlabel_df.pickle')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load the model
model_conv = torchvision.models.resnet50(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_ftrs = model_conv.fc.in_features
nb_classes = len(class_names)
model_conv.fc = nn.Sequential(nn.Dropout(0.1),nn.Linear(num_ftrs, nb_classes))
model_conv = model_conv.to(device)
modelfolder = '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/models/resnet50/multilabel/'
confusion_dfs = os.listdir(conffolder)
models = os.listdir(modelfolder)

model = models[0]
model_conv.load_state_dict(torch.load(modelfolder + model))
model_conv = model_conv.to(device)

def visualizeCAMs_forClass_forFold(model_conv, modeltype, class_name):
    test_IDs = []
    test_labels = []
    for pid, label in zip(labels_df.pid, labels_df.label):
        if label == class_name:
            test_IDs.append(pid)
            test_labels.append(label)


    test_ds = TiledArgusDS.TiledArgusTestDS(basedir, test_IDs, 1024, transform = test_transform)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1) #num workers is how many subprocesses to use

    plotdir = plotfolder + '/{0:s}_class{1:s}/'.format(modeltype, class_name)
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    visualize_model_withCAM(model_conv, plotdir, class_names, class_name, test_dl)

visualizeCAMs_forClass_forFold(model_conv,'multilabel', 'E')

testclass = np.unique(labels_df.label)
conf_dtname = [cc for cc in confusion_dfs if 'foldno{}'.format(split_no) in cc][0]
conf_df = pd.read_pickle(conffolder + conf_dtname)
prob_df = calc_prob(conf_df)

fig, ax = plt.subplots(10,2, sharex = True, sharey = True, tight_layout = True)
ax.ravel('F')[5].set_xlim(0, len(class_names))
ax.ravel('F')[5].set_xticks(np.arange(len(class_names)))
ax.ravel('F')[9].set_xticks(np.arange(len(class_names)))
ax.ravel('F')[5].set_xticklabels(class_names)
ax.ravel('F')[9].set_xticklabels(class_names)
fig.suptitle(model)
fig.set_size_inches(6.5,12)

for ti,tt in enumerate(testclass):
    if 'SSZ' in tt:
        continue
    test_IDs = []
    test_labels = []
    for pid, label in zip(labels_df.pid, labels_df.label):
        if label == tt:
            test_IDs.append(pid)
            test_labels.append(label)


    test_ds = ArgusDS.ArgusTestDS(basedir, test_IDs, transform = test_transform)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1) #num workers is how many subprocesses to use

    dataset_sizes = {'test':len(test_ds)}


    probs = test_model(model_conv, test_dl)
    meanprobs =  np.mean(probs, axis = 0)
    std = np.std(probs, axis = 0)


    ax.ravel('F')[ti].set_title("Given class {}".format(tt), fontsize = 10)
    ax.ravel('F')[ti].bar(np.arange(len(meanprobs)),meanprobs, width = 0.5, yerr = std )
    ax.ravel('F')[ti].set_ylim(0,1)
    ax.ravel('F')[ti].set_xlim(0,len(class_names))

    if tt in prob_df.index:
        ax.ravel('F')[ti].plot(np.arange(len(meanprobs)), prob_df.loc[tt].values)
        ax.ravel('F')[ti].scatter(np.arange(len(meanprobs)), prob_df.loc[tt].values)


plt.savefig(plotfolder + 'NoTiledClassMeansandStdFoldno{}.png'.format(split_no))








