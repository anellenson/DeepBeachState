from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import pickle
import eofs

cross_cut = 50
mean_allcams = {}
states = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
color = {'nbn':'magenta', 'duck':'darkblue'}
fig_intensity, ax_i = pl.subplots(1,1)
for trainsite in ['duck']:
    fig_eof, ax_eof = pl.subplots(5,2,figsize = [3,9])
    fig_avg, ax_avg = pl.subplots(5, 1, figsize = [5,9])
    i_on = []
    i_off = []
    for i, state in enumerate(states):
            modelname = 'resnet512_five_aug_0'
            testsite = 'nbn'
            vis_dir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/model_output/train_on_{}/{}/visualize/test_on_{}/'.format(trainsite, modelname, testsite)


            all_cams = np.empty((15, 512, 512))
            for ii in [1, 5, 10]:
                with open(vis_dir + 'BackCam_{}_{}.pickle'.format(state, ii), 'rb') as f:
                    cams = pickle.load(f)

                cams[cams<0] = 0
                all_cams[ii-1:ii-1+5,:,:] = cams

            solver = eofs.standard.Eof(all_cams)

            eofs_outs = solver.eofs(neofs = 2)

            meancams = np.nanmean(all_cams, axis = 0)
            mean_allcams[trainsite] = meancams
            intensity_onshore = mean_allcams[trainsite][:cross_cut,:].sum()/np.sum(meancams)
            intensity_offshore = mean_allcams[trainsite][cross_cut:,:].sum()/np.sum(meancams)


            i_on.append(intensity_onshore)
            i_off.append(intensity_offshore)
            print('percentage intensity off shore for {} for {}'.format(intensity_offshore, trainsite))
            print('percentage intensity on shore for {} for {}'.format(intensity_onshore, trainsite))

            ax_avg[i].set_title('{}'.format(state))
            ax_avg[i].imshow(mean_allcams[trainsite], cmap = 'hot')
            ax_avg[i].plot((0, 510), (cross_cut, cross_cut), color  =  'w')

            ax_eof[i,0].set_title('{}'.format(state))
            ax_eof[i,0].imshow(eofs_outs[0], cmap = 'hot')
            ax_eof[i,1].imshow(eofs_outs[1], cmap = 'hot')


    ax_i.plot(range(len(states)), i_on, '-*', color = color[trainsite], label = '{} Int Onshore'.format(trainsite))
    ax_i.plot(range(len(states)), i_off, '--o', color = color[trainsite], label = '{} Int Offshore'.format(trainsite))

    fig_eof.suptitle('EOFs Tested at {}, trained at {}'.format(testsite, trainsite))
    fig_avg.suptitle('AVGs Tested at {}, trained at {}'.format(testsite, trainsite))

    fig_avg.savefig(vis_dir + 'state_avg.png')
    fig_eof.savefig(vis_dir + 'state_eof.png')

ax_i.legend()
ax_i.set_xticks(range(len(states)))
ax_i.set_xticklabels(states)
ax_i.set_ylabel('Percentage of Intensity')
