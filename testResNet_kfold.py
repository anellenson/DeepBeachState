import pydevd_pycharm
pydevd_pycharm.settrace('149.171.148.109', port=22, stdoutToServer=True, stderrToServer=True)

from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as pl
import ArgusDS
import pickle
import pandas as pd
import preResnet as pre
import os
import cv2
import scipy.io as sio
import postResnet as post
import pickle
from collections import Counter
from torch.autograd import Variable
from CAMplot import CAMplot

######This will check the accuracy vs probability

#Configurations
pl.ion()
plot_CAMs = True
#Resolution for tiled ds, res1/res2 for nontiled
resolution = 512
res2 = 512
mean = 0.48
std = 0.29 #This is from the 'calc mean'
modeltype = 'oblique'
class_names = ['Ref','LTT/B','TBR/CD','RBB/E','LBT/FG']
pid_df = 'labels/pure_label_df.pickle'
labels_df = pd.read_pickle(pid_df)
pids, labels, labels_dict = pre.createLabelsDict(labels_df, class_names)
no_class = pre.countClasses(labels_dict)
weights = [1/np.cbrt(no) for no in no_class]
class_weights = torch.FloatTensor(weights).cuda()
prob_votes = False #This is to turn on the calculation of probabilities from votes
prob_seg = True

#This is the plot name of the accuracy figure
modelname = 'pure_images_h512_w512_more_labels_run3'
CAMplotdir = '/plots/' + modelname + '/'
modeltype = 'notiled' #tiled or no tiled
imagedir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/rect/test/'
modelfolder = 'resnet_models/best_models/'
multilabel_bool = False
nbn = False

'''
kfold_file = 'kfold_pure_images.mat'
with open(kfold_file, 'rb') as file:
    kfold_dict = pickle.load(file)

with open('labels/prob_seg_dict.pickle','rb') as dict_pick:
    prob_pid_dict = pickle.load(dict_pick)
'''

all_files = os.listdir(imagedir)
test_IDs = [file for file in allfiles if 'synthetic_states' in file]

#labels_df = pd.DataFrame({'pid':list(prob_pid_dict.keys()), 'label':list(prob_pid_dict.values())})
###Info about the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Load the model
model_conv = torchvision.models.resnet50(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_ftrs = model_conv.fc.in_features
nb_classes = len(class_names)
model_conv.fc = nn.Sequential(nn.Dropout(0.1),nn.Linear(num_ftrs, nb_classes))
model_conv = model_conv.to(device)

test_transform = transforms.Compose([transforms.Resize((resolution,res2)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                        transforms.Normalize([mean, mean, mean],[std, std, std]),
                                ])

def calc_prob(conf_dt):
    total_perclass = np.sum(conf_dt, axis = 1)
    prob_perclass = conf_dt.values/total_perclass.values
    prob_df = pd.DataFrame(data = prob_perclass, columns = conf_dt.columns, index = conf_dt.index)
    return prob_df

def test_model(model, dataloader, multilabel =False):

    model.eval()   # Set model to evaluate mode

    totalpreds = []
    totalprobs = []
    totalvotes = []
    testpids = []
    allCAMs = []
    testinps = []

    with torch.no_grad():
        for i, (inputs, pid) in enumerate(dataloader):
            if plot_CAMs: #Register a forward hook if you want to plot cams
                features_blobs = []

                def hook_feature(module, input, output):
                    features_blobs.append(output.data.cpu().numpy())

                model._modules.get('layer4').register_forward_hook(hook_feature)

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
                    pl.pause(0.001)

            pid = pid[0]
            testpids.append(pid)
            testinps.append(inputs)
            inputs = inputs.to(device)
            outputs = model(inputs)
            if multilabel_bool:
                #This will return a multilabel prediction
                probs = torch.nn.functional.softmax(outputs)
                out_sigmoid = torch.sigmoid(outputs)
                t = Variable(torch.Tensor([0.5])).cuda()  # establish threshold
                preds = (out_sigmoid > t).float() * 1
                totalpreds.append(preds.cpu().numpy()[0])

            else:
                #This will return one prediction
                probs = torch.nn.functional.softmax(outputs)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()[0]
                totalpreds.append(preds)

            probs = probs[0].cpu().numpy()
            #Find top probabilities
            top_probs = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)

            #Load probabilities from the pid_dictoinary
            if pid_votes:
                try:
                    votes = prob_pid_dict[pid]
                    hist_votes = Counter(votes)
                    #Sort the probabilities according to where they are in the class list
                    sorted_votes = np.zeros((len(probs),1))
                    for key in list(hist_votes.keys()):
                        if key in class_names:
                            sorted_votes[class_names.index(key)] = hist_votes[key]


                    #Turn into probabilties
                    prob_votes = sorted_votes/np.sum(sorted_votes)
                    totalvotes.append(prob_votes)
                    #multiply by the probability of getting it right
                    if multilabel_bool:
                        #Generate multilabel
                        multi_target = np.zeros((len(class_names),))
                        multi_target[np.where((sorted_votes != 0))[0]] = 1

                except KeyError:
                    pass

            if plot_CAMs:
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

                CAMs = returnCAM(features_blobs[0], weight_softmax, top_probs)
                #Save out cams that belong in certain categories:
                allCAMs.append(CAMs[0])

            totalprobs.append(probs)

            if i % 100 == 0:
                print("On iteration {}".format(i))

    #Recast the probabilities into an array
    totalprobs_array = np.empty((len(totalprobs), len(probs)))
    totalvotes_array = np.empty((len(totalprobs), len(probs)))
    for ti,probs in enumerate(totalprobs):
        totalprobs_array[ti,:] = probs

    for ti,votes in enumerate(totalvotes):
        totalvotes_array[ti,:] = votes.squeeze()


    return totalprobs_array, totalpreds, totalvotes_array, testpids, allCAMs, testinps

'''
fig, ax = pl.subplots(10,2, sharex = True, sharey = True, tight_layout = True)
ax.ravel('F')[5].set_xlim(0, len(class_names))
ax.ravel('F')[5].set_xticks(np.arange(len(class_names)))
ax.ravel('F')[9].set_xticks(np.arange(len(class_names)))
ax.ravel('F')[5].set_xticklabels(class_names)
ax.ravel('F')[9].set_xticklabels(class_names)
fig.suptitle(model)
fig.set_size_inches(6.5,12)
'''
models = os.listdir(modelfolder)
models.sort() #Sort so they correspond with the train/test splits which will be called by index
with open(kfold_file, 'rb') as file:
    kfold_dict = pickle.load(file)

kfold_testfiles = kfold_dict['testfiles']
#This is for a given class, what's the accuracy

meanprobs_model = np.zeros((len(models),len(class_names)))
state_std_model = np.zeros((len(models),len(class_names)))
P_P_sum = np.zeros((len(class_names), len(class_names)))
split_no = 0
model = [mm for mm in models if 'foldno{}'.format(str(split_no)) in mm]
model = model[0]

##Load the labels that were in the validation set for each class
if modeltype == 'tiled':
    print("Importing tiled dataset")
    import TiledArgusDS
    test_ds = TiledArgusDS.TiledArgusTestDS(basedir, test_IDs, nbn = nbn, resolution, transform = test_transform)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1) #num workers is how many subprocesses to use
else:
    import ArgusDS
    test_ds = ArgusDS.ArgusTestDS(basedir, test_IDs, nbn = nbn, transform = test_transform)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1) #num workers is how many subprocesses to use


model_conv.load_state_dict(torch.load(modelfolder + model))
model_conv = model_conv.to(device)

totalprobs, totalpreds, totalvotes, testpids, allCAMs, testinps = test_model(model_conv, test_dl, multilabel = multilabel_bool)
results_dict = {'totalprobs':totalprobs, 'CNNpreds'}


Pdot = 0
for probs, prob_votes in zip(totalprobs, totalvotes):
    Pdot = Pdot + np.dot(probs, prob_votes)
    P_P = np.outer(probs, prob_votes)
    P_P_sum = P_P_sum + P_P

if plot_CAMs:
    bins = [0, 0.2, 0.4, 0.6, 0.8]
    CAMplot = CAMplot(totalprobs, totalpreds, totalvotes, testpids, allCAMs, testinps, class_names)
    state_and_prob_binned_dict = CAMplot.bin_CAMs(bins)
    CAMplot.find_and_plot_binned_mean_var(state_and_prob_binned_dict, bins, resolution, CAMplotdir)
    #CAMplot.plot_individual_CAMs(CAMplotdir)

#Now find accuracy, confusion table
if not multilabel_bool:
    confusion_matrix = np.zeros((len(class_names), len(class_names)))
    for cnn_prediction,votes_for_image in zip(totalpreds, totalvotes):
        true_class = np.where((votes_for_image == np.max(votes_for_image))) #insert here new for the 'segmented' image
        confusion_matrix[true_class, cnn_prediction] += 1

    confusion_matrix = (confusion_matrix.T/np.sum(confusion_matrix, axis = 1)).T

    confplotname = '/plots/model_name/conftable_{}.{}.png'.format(modelname,split_no)
    fig, ax = pl.subplots(1,1)
    cl = ax.pcolor(np.flipud(confusion_matrix), cmap = 'Reds')
    for ai, acc in enumerate(confusion_matrix.diagonal()):
        ax.text(ai+0.15, 7 - ai + 0.5, '{0:.2f}'.format(acc), fontweight = 'bold')
    pl.colorbar(cl)
    cl.set_clim((0,1))
    ax.set_xlabel('Confused As')
    ax.set_ylabel('CNN')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(np.arange(len(class_names[::-1])))
    ax.set_yticklabels(class_names[::-1])
    ax.set_title(modelname + ' Normalized Confusion Matrix Split {}, Pdot of {}'.format(split_no, Pdot))
    pl.savefig(confplotname)

#normalize the PP
P_P_norm = (P_P_sum.T/np.sum(P_P_sum, axis = 1)).T

fig, ax = pl.subplots(1,1)
cl = ax.pcolor(np.flipud(P_P_norm),cmap = 'Blues')
for ai, acc in enumerate(P_P_norm.diagonal()):
    ax.text(ai+0.15, 7 - ai + 0.5, '{0:.2f}'.format(acc), fontweight = 'bold')
pl.colorbar(cl)
cl.set_clim((0,1))
ax.set_xlabel('Human Probabilities')
ax.set_ylabel('CNN Probabilities')
ax.set_xticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticks(np.arange(len(class_names[::-1])))
ax.set_yticklabels(class_names)
ax.set_title(modelname + ' P*P')
pl.savefig(PPplotname)


