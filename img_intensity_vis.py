from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import pickle
import pandas as pd
import seaborn as sns


cross_cut = 50
mean_allcams = {}
states = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
color = {'nbn':'magenta', 'duck':'darkblue'}

for trainsite in ['duck', 'nbn']:

    fig_avg, ax_avg = pl.subplots(5, 1, figsize = [5,9])

    for i, state in enumerate(states):
        modelname = 'resnet512_five_aug_trainloss_0'
        testsite = 'nbn'
        vis_dir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/model_output/train_on_{}/{}/visualize/test_on_{}/'.format(trainsite, modelname, testsite)

        Icam_dict = {}
        all_Icams = np.empty((25, 512))
        camnum = 0
        clms = []

        for ii in [1, 5, 10, 15, 20]:
            with open(vis_dir + 'BackCam_{}_{}.pickle'.format(state, ii), 'rb') as f:
                cams = pickle.load(f)

            for cam in enumerate(cams):
                cam_xs = np.mean(cam[1], axis = 1)
                Icam = np.zeros((len(cam_xs)))
                camnum += 1

                for jj in range(len(cam_xs)):
                    Icam[jj] = cam_xs[:jj].sum()/cam_xs.sum()

                all_Icams[camnum] = Icam

        Icam_dict.update({state:all_Icams.ravel()})

columns = states + ['test_site'] + ['xpt']
Icams_df = pd.DataFrame(data = Icam_dict)
Icams_df['xpt'] = range(512)*25
Icams_df['test_site'] =


    ax_avg[i].set_title('{}'.format(state))
    ax_avg[i].imshow(mean_allcams[trainsite], cmap = 'hot')
    ax_avg[i].plot((0, 510), (cross_cut, cross_cut), color  =  'w')
