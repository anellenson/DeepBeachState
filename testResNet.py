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
from ResultPlot import TestResultPlot
from scipy.stats import entropy

######This will check the accuracy vs probability

#Configurations
pl.ion()
plot_CAMs = False
#Resolution for tiled ds, res1/res2 for nontiled
removed_shoreline_mean = [0.2714, 0.3129, 0.3416]
removed_shoreline_std = [0.3037, 0.3458, 0.3769]
rgb_mean = [0.4066, 0.4479, 0.4734]
rgb_std = [0.2850, 0.3098, 0.3322]
removed_shoreline_histeq_mean = [0.1680, 0.1719, 0.1871]
removed_shoreline_histeq_std = [0.2567, 0.2591, 0.2708]
duck_gray_mean = [0.5199, 0.5199, 0.5199]
duck_gray_std = [0.2319, 0.2319, 0.2319]

channels_mean = duck_gray_mean
channels_std = duck_gray_std
class_names = ['Ref','LTT-B','TBR-CD','RBB-E','LBT-FG']
#class_names = ['B','C','D','E','F','G','Calm','NoVis']
test_site = 'nbn'
train_site = 'duck'

pid_df_dict = {'duck':'labels/duck_daytimex_labels_df.pickle', 'nbn':'nbn_labels_cleaned_165.pickle'}
pid_df = pid_df_dict[test_site]
labels_df = pd.read_pickle(pid_df)
pid_votes = False #This is to turn on the calculation of probabilities from votes
simplex_labels = False
regression = False
conf_outdir = 'confusion_table_results/{}/train_on_{}/'.format(test_site, train_site)
if not os.path.exists(conf_outdir):
    os.mkdir(conf_outdir)

#This is the plot name of the accuracy figure
modelfolder = 'resnet_models/train_on_{}/'.format(train_site) #this is the training folder
modelnames = os.listdir(modelfolder)

model_out_dir = 'model_output/{}/train_on_{}'.format(test_site,train_site)
if not os.path.exists(model_out_dir):
    os.mkdir(model_out_dir)

for modelname in modelnames:
    if not os.path.exists('plots/{}/{}'.format(test_site,modelname)):
        os.mkdir('plots/{}/{}'.format(test_site,modelname))
    CAMplotdir = 'plots/{}/{}/CAMs/'.format(test_site,modelname)
    if not os.path.exists(CAMplotdir):
        os.mkdir(CAMplotdir)

    basedirs = ['/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/orig_gray/',
                '/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn/']
    multilabel_bool = False
    with open('labels/{}_valfiles_15perclass.pickle'.format(test_site), 'rb') as f:
        valfiles = pickle.load(f)
    test_IDs = valfiles['valfiles']
    res_height = 512  # height
    res_width = 512  # width

    ####Load the model / info about the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_conv = torchvision.models.resnet50(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_ftrs = model_conv.fc.in_features
    nb_classes = len(class_names)
    model_conv.fc = nn.Linear(num_ftrs, nb_classes)
    model_conv = model_conv.to(device)

    intermediate_layers = {'layer1':[], 'layer2':[], 'layer3':[], 'layer4':[]}

    test_transform = transforms.Compose([transforms.Resize((res_height, res_width)),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize(channels_mean,channels_std),
                                    ])

    def test_model(model, dataloader, multilabel =False):

        model.eval()   # Set model to evaluate mode

        totalpreds = []
        totalprobs = []
        simplex_truth = []
        testpids = []
        allCAMs = []
        testinps = []
        allweights = []

        with torch.no_grad():
            for i, (inputs, pid) in enumerate(dataloader):
                if plot_CAMs: #Register a forward hook if you want to plot cams

                    features_blobs1 = []
                    features_blobs2 = []
                    features_blobs3 = []
                    features_blobs4 = []
                    def hook_feature_l1(module, input, output):
                        output = output.cpu().data.numpy()
                        features_blobs1.append(output)

                    def hook_feature_l2(module, input, output):
                        output = output.cpu().data.numpy()
                        features_blobs2.append(output)

                    def hook_feature_l3(module, input, output):
                        output = output.cpu().data.numpy()
                        features_blobs3.append(output)

                    def hook_feature_l4(module, input, output):
                        output = output.cpu().data.numpy()
                        features_blobs4.append(output)

                    model._modules.get('layer4').register_forward_hook(hook_feature_l4)
                    model._modules.get('layer3').register_forward_hook(hook_feature_l3)
                    model._modules.get('layer2').register_forward_hook(hook_feature_l2)
                    model._modules.get('layer1').register_forward_hook(hook_feature_l1)



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
                if plot_CAMs:
                    intermediate_layers['layer4'].append(features_blobs4[0])
                    intermediate_layers['layer3'].append(features_blobs3[0])
                    intermediate_layers['layer2'].append(features_blobs2[0])
                    intermediate_layers['layer1'].append(features_blobs1[0])

                if multilabel_bool:
                    if regression:
                    #This will return a multilabel prediction
                        totalpreds.append(np.nan)
                        probs = outputs
                    else:
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
                top_probs = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)[:3]

                #Load probabilities from the pid_dictoinary
                if pid_votes:
                    try:
                        votes = prob_pid_dict[pid] #this is in a form of a dictionary of votes
                        hist_votes = Counter(votes)
                        #Sort the probabilities according to where they are in the class list
                        sorted_votes = np.zeros((len(probs),1))
                        for key in list(hist_votes.keys()):
                            if key in class_names:
                                sorted_votes[class_names.index(key)] = hist_votes[key]


                        #Turn into probabilties
                        prob_votes = sorted_votes/np.sum(sorted_votes)
                        simplex_truth.append(prob_votes)
                        #multiply by the probability of getting it right
                        if multilabel_bool:
                            #Generate multilabel
                            multi_target = np.zeros((len(class_names),))
                            multi_target[np.where((sorted_votes != 0))[0]] = 1

                    except KeyError:
                        pass

                if simplex_labels:
                    simplex = simplex_df.loc[pid].values[:len(class_names)]
                    simplex_truth.append(simplex)

                if plot_CAMs:
                    params = list(model.parameters())
                    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

                    def returnCAM(feature_conv, weight_softmax, class_idx):
                        # generate the class activation maps upsample to 256x256
                        size_upsample = (res_height, res_width)
                        bz, nc, h, w = feature_conv.shape
                        output_cam = []
                        for bb in np.arange(bz):
                            for idx in class_idx[0:1]:
                                cam = weight_softmax[idx].dot(feature_conv[bb].reshape((nc, h*w)))
                                cam = cam.reshape(h, w)
                                cam = cam - np.min(cam)
                                cam_img = cam / np.max(cam)
                                cam_img = np.uint8(255 * cam_img)
                                output_cam.append(cv2.resize(cam_img, size_upsample))
                        return output_cam

                    CAMs = returnCAM(features_blobs4[0], weight_softmax, top_probs)
                    #Save out cams that belong in certain categories:
                    allCAMs.append(CAMs)
                    allweights.append(weight_softmax)

                totalprobs.append(probs)

                if i % 100 == 0:
                    print("On iteration {}".format(i))

        #Recast the probabilities into an array
        CNN_probs = np.empty((len(totalprobs), len(probs)))
        human_probs = np.empty((len(totalprobs), len(probs)))
        for ti,probs in enumerate(totalprobs):
            CNN_probs[ti,:] = probs

        for ti,simplex in enumerate(simplex_truth):
            human_probs[ti,:] = simplex


        return CNN_probs, totalpreds, human_probs, testpids, allCAMs, testinps, allweights, intermediate_layers


    test_ds = ArgusDS.ArgusTestDS(basedirs, test_IDs, transform = test_transform)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1) #num workers is how many subprocesses to use

    model_conv.load_state_dict(torch.load(modelfolder + modelname))
    model_conv = model_conv.to(device)

    CNNprobs, totalpreds, humanprobs, testpids, allCAMs, testinps, allweights, intermediate_layers = test_model(model_conv, test_dl, multilabel = multilabel_bool)

    model_out = {'CNNprobs': CNNprobs, 'testpids':testpids}
    with open(model_out_dir + '/' + modelname[:-4] + '_CNNprobs_testpids.pickle', 'wb') as f:
        pickle.dump(model_out, f)



# f = open('model_output/trained_on_duck_tested_on_nbn_run{}.pickle'.format(run),'wb')
# pickle.dump(preds, f, protocol = 0)
# f.close()

#
# if plot_CAMs:
#     #Don't plot ALL the cams, just plot the individual ones where the CNN does the best and the worst
#     bins = [0, 0.4, 0.6, 0.8, 1]
#     CAMplot = CAMplot(CNNprobs, totalpreds, humanprobs, testpids, allCAMs, testinps, class_names)
#     state_and_prob_binned_dict = CAMplot.bin_CAMs(bins)
#     CAMplot.find_and_plot_binned_mean_var(state_and_prob_binned_dict, bins, resolution, res2, CAMplotdir)
#     CAMplot.plot_one_CAM(CAMplotdir, np.arange(len(testpids[:10])))
#     print("plotting CAMs")
#     # for metric, metric_name in zip([Pdot, distance, KLdivergence], ['Pdot', 'distance', 'KLdivergence']):
#     #    CAM_inds = plotter.top_and_bottom_quartiles(metric)
#     #   if not os.path.exists(CAMplotdir + '/' + metric_name):
#     #      os.mkdir(CAMplotdir + '/' + metric_name)
#     #
#     #    CAMplot.plot_individual_CAMs(CAMplotdir + '/' + metric_name + '/', CAM_inds)

