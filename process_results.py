import pandas as pd
import os
import matplotlib.pyplot as pl
import numpy as np
import torchvision
import postResnet as post
import plotTools as pt
import pickle
import torch
import torch.nn as nn

def calc_mean(conf_dt):
    right = np.diagonal(conf_dt.values)
    class_acc = right/np.sum(conf_dt, axis =1)
    meanclassacc = np.mean(class_acc)
    all_acc = np.append(class_acc, meanclassacc)
    return all_acc

def load_df(dirname, classes):
    files = os.listdir(dirname)
    meanclasses = [cc for cc in classes]
    meanclasses.append('meanacc')
    results_df = pd.DataFrame(columns = meanclasses, index = files)
    for ff in files:
        try:
            conf_dt = pd.read_pickle(dirname + ff)
        except IOError:
            continue
        all_acc = calc_mean(conf_dt)
        results_df.loc[ff][:] = all_acc

    results_df = results_df.sort_values(by = 'meanacc',ascending = False)
    return results_df

for runtype in ['duck', 'nbn', 'nbn_duck']:

    with open('model_output/{}/train_full_ResNet_{}.pickle'.format(runtype, runtype), 'rb') as f:
        trainInfo = pickle.load(f)

    plot_fname = 'plots/run_info_augmented_images_{}'.format(runtype)
    pt.trainInfo(trainInfo['val_acc'], trainInfo['train_acc'], trainInfo['val_loss'], trainInfo['train_loss'], plot_fname, runtype)


