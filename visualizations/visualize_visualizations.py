import seaborn as sns
import numpy as np
from matplotlib import pyplot as pl
import pickle
import pandas as pd
import time
from PIL import Image
import VisTools as vt

def img_votes(trainsite, testsite, model_basename):
    runno = 0
    while runno<10:
        modelname = model_basename + '_{}'.format(runno)
        preds_pickle = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/model_output/train_on_{}/{}/cnn_preds.pickle'.format(trainsite, modelname)

        with open(preds_pickle, 'rb') as f:
            preds = pickle.load(f)

        valfiles = preds['{}_valfiles'.format(testsite)]
        cnn_preds = preds['{}_CNN'.format(testsite)]
        truth = preds['{}_truth'.format(testsite)]
        if runno == 0:
            master_pd = pd.DataFrame(index = valfiles, data = {'truth':truth, 'preds_{}'.format(runno):cnn_preds, 'testsite':testsite})
        if runno > 0:
            valfiles_pd = pd.DataFrame(index = valfiles, data = {'preds_{}'.format(runno):cnn_preds})
            master_pd = master_pd.join(valfiles_pd, how = 'inner')

        runno +=1

    return master_pd

def img_accuracies(master_pd):

    for pid in master_pd.index:
        num_right = 0
        for runno in range(10):
            cnnval = master_pd.loc[pid]['preds_{}'.format(runno)]
            if cnnval == master_pd.loc[pid]['truth']:
                num_right += 1
        acc = num_right/10

        master_pd.loc[master_pd.index == pid, 'acc'] = acc

    return master_pd

def link_xs_pts(full_img_pd):

    with open('xs_pts.pickle', 'rb') as f:
        xs_pts = pickle.load(f)

    for pid, (xpt1, xpt2) in xs_pts.items():
        testsite = full_img_pd.loc[pid]['testsite']
        img = Image.open(imgdirs[testsite] + pid)
        img = img.resize((512,512))
        img_array = np.array(img)
        transect = img_array[:,256]


        full_img_pd.loc[pid, 'x1'] = xpt1
        full_img_pd.loc[pid, 'x2'] = xpt2

        I1 = transect[xpt1]
        I2 = transect[xpt2]

        full_img_pd.loc[pid, 'I1'] = I1
        full_img_pd.loc[pid, 'I2'] = I2

        I_ratio = I2/I1

        full_img_pd.loc[pid, 'Iratio'] = I_ratio


    return full_img_pd



master_pd_duck = img_votes('nbn_duck', 'duck', 'resnet512_five_aug_trainloss')
master_pd_nbn = img_votes('nbn_duck', 'nbn', 'resnet512_five_aug_trainloss')
master_pd_duck = img_accuracies(master_pd_duck)
master_pd_nbn = img_accuracies(master_pd_nbn)
master_pd_duck['testsite'] = 'duck'
master_pd_nbn['testsite'] = 'nbn'
full_img_pd = pd.concat((master_pd_duck, master_pd_nbn))

imgdirs = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/',
           'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}

states = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
imgtype = 'resize_128'
variog = vt.Variograms()

img_variog0, img_variog90, lag0, lag90 = variog.load_variograms()
colors = {'nbn': 'r', 'duck':'b'}

for testsite in ['nbn', 'duck']:
    full_lag0 = list(lag0*7.03) * 25
    full_lag90 = list(lag90*2.34) * 25

    fig, ax = pl.subplots(5,2, sharey = True)
    fig_avg, ax_avg = pl.subplots(5,2, sharey = True)
    for state in range(len(states)):
            variog0 = []
            variog90 = []
            variog0_df, variog90_df = variog.average_variograms(state, full_img_pd, testsite, img_variog0, img_variog90)
            variog0_df['lag'] = lag0*7.03
            variog90_df['lag'] = lag90*2.34
            for imnum in range(25):
                sns.lineplot(x = 'lag', y = 'varpt{}'.format(imnum), data = variog0_df, ax = ax[state,0])
                sns.lineplot(x = 'lag', y = 'varpt{}'.format(imnum), data = variog90_df, ax = ax[state,1])
                variog0 += list(variog0_df['varpt{}'.format(imnum)].values)
                variog90 += list(variog90_df['varpt{}'.format(imnum)].values)

            ax[state, 0].set_ylabel(states[state])
            ax[state,0].plot((0, lag0[-2]*7.03),(1,1), '--k')
            ax[state,1].plot((0, lag90[-2]*2.34),(1,1), '--k')

            full_variog0_df = pd.DataFrame({'lag':full_lag0, 'varpts':variog0})
            full_variog90_df = pd.DataFrame({'lag':full_lag90, 'varpts':variog90})

            sns.lineplot(x = 'lag', y = 'varpts', data = full_variog0_df, ax = ax_avg[state,0])
            sns.lineplot(x = 'lag', y = 'varpts', data = full_variog90_df, ax = ax_avg[state,1])

            ax_avg[state, 0].set_ylabel(states[state])
            ax_avg[state,0].plot((0, lag0[-2]*7.03),(1,1), '--k')
            ax_avg[state,1].plot((0, lag90[-2]*2.34),(1,1), '--k')


            # ax[state,0].set_xticklabels([])
            # ax[state,1].set_xticklabels([])

    ax[0,0].set_title('Azimuth 0 deg')
    ax[0,1].set_title('Azimuth 90 deg')
    ax[state,0].set_xlabel('lag (m)')
    ax[state,1].set_xlabel('lag (m)')

    ax_avg[0,0].set_title('Azimuth 0 deg')
    ax_avg[0,1].set_title('Azimuth 90 deg')
    ax_avg[state,0].set_xlabel('lag (m)')
    ax_avg[state,1].set_xlabel('lag (m)')

    fig.suptitle('Image Variograms at {}'.format(testsite))
    fig_avg.suptitle('Image Variograms at {}'.format(testsite))

    vis_dir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/visualizations'
    manuscript_plot_dir = '/home/aquilla/aellenso/Research/DeepBeach/resnet_manuscript/'
    fig.savefig(vis_dir + '/plots/overall_variograms_allimgs_{}.png'.format(testsite))
    fig_avg.savefig(vis_dir + '/plots/avg_variograms_allimgs_{}.png'.format(testsite))

full_img_pd = variog.variogram_chars_df(img_variog0, img_variog90, lag0, lag90, full_img_pd)

full_img_pd = link_xs_pts(full_img_pd)

metrics = [pp for pp in full_img_pd.keys() if '0' in pp and '90' not in pp and 'preds' not in pp]
metrics = [pp for pp in full_img_pd.keys() if '90' in pp and 'preds' not in pp]
fig, axes = pl.subplots(7,2)
for mi, metric in enumerate(metrics):
    ax = axes.ravel('F')[mi]
    g = sns.boxplot(x = 'truth', y = metric,  data = full_img_pd, hue = 'testsite', ax = ax)
    ax.legend_.remove()

fig, axes = pl.subplots(2,1, sharex = True, tight_layout = {'rect':[0, 0, 1, 0.95]})
for mi, metric in enumerate(['FDO']):
    ax = axes[mi]
    metricplot = metric + '0'
    sns.boxplot(x= 'truth', y = metricplot, data = full_img_pd, hue = 'testsite', ax = ax,  palette = {'both':'black', 'nbn':'salmon', 'duck':'blue'})
    ax.legend_.remove()
    ax.set_xlabel('')
    ax.set_ylabel('FDO-CS')
    metricplot = metric + '90'
    ax = axes[mi+1]
    sns.boxplot(x= 'truth', y = metricplot, data = full_img_pd, hue = 'testsite', ax = ax,  palette = {'both':'black', 'nbn':'salmon', 'duck':'blue'})
    ax.legend_.remove()
    ax.set_xlabel('')
    ax.set_ylabel('FDO-AS')

axes[1].set_xlabel('Beach State')
axes[1].set_xticklabels(states)
fig.suptitle('Variogram Metrics')
fig.savefig(manuscript_plot_dir + 'plots/fig8_variog_metrics.png')

for testsite in ['nbn', 'duck']:
    for statenum in range(5):
        fig, ax = pl.subplots(3,3)

        for bi, percentile in enumerate([(0, 0.5), (0.5, 0.8), (0.8, 1)]):

            pids = full_img_pd[(full_img_pd.testsite == testsite) &(full_img_pd.truth == statenum) & (full_img_pd.acc >= percentile[0]) & (full_img_pd.acc < percentile[1])].index
            binned_trans = np.zeros((len(pids), 512))
            binned_var0 = np.zeros((len(pids), 16))
            binned_var90 = np.zeros((len(pids), 46))


            for pi, pid in enumerate(pids):
                img = Image.open(imgdirs[testsite] + pid)
                img = img.resize((512,512))
                var0 = img_variog0[pid]
                var90 = img_variog90[pid]
                img = np.array(img)
                transect = img[:, 256]

                binned_trans[pi, :] = transect
                binned_var0[pi, :] = var0[:-1]
                binned_var90[pi, :] = var90[:-1]

            pts = list(range(512)) * len(pids)

            trans_df = pd.DataFrame(data = {'transect':binned_trans.flatten(), 'pts':pts})
            var0_df = pd.DataFrame(data = {'var0':binned_var0.flatten(), 'lag0':list(lag0[:-1])*len(pids)})
            var90_df = pd.DataFrame(data = {'var90':binned_var90.flatten(), 'lag90':list(lag90[:-1])*len(pids)})

            sns.lineplot(y = 'transect', x = 'pts', data = trans_df, ax = ax[bi,0])
            sns.lineplot(y = 'var0', x = 'lag0', data = var0_df, ax = ax[bi, 1])
            sns.lineplot(y = 'var90', x = 'lag90', data = var90_df, ax = ax[bi, 2])

            fig.suptitle('state {} Site {}'.format(statenum, testsite))



for si, state in enumerate(states):
    sns.lineplot(x = 'pixelpts', y = state, hue = 'testsite', data = transects_pd, ax = ax[si])
    ax[si].set_xlim((0, 400))



















full_img_pd = variog.variogram_chars_df(img_variog0, img_variog90, lag0, lag90, full_img_pd)
pl.figure()
sns.scatterplot(x = 'RMM0', y = 'acc', hue = 'testsite', data = full_img_pd)

metrics = [name for name in full_img_pd.columns if '0' in name and 'preds' not in name and '90' not in name]
fig, axes = pl.subplots(7,2, tight_layout= True)
for metric in metrics:
    fig, axes = pl.subplots(5,1)
    for statenum in range(5):
        ax = axes[statenum]
        sns.scatterplot(x = metric, y = 'acc', data = full_img_pd[full_img_pd.truth == statenum], hue = 'testsite', ax = ax, legend = False, color = 'blue')


fig, ax = pl.subplots(1,1)
sns.scatterplot(x = 'RMM90', y = 'acc', data = master_pd_duck, hue = 'truth', marker = 'x')
sns.scatterplot(x = 'RMM90', y = 'acc', data = master_pd_nbn, hue = 'truth', marker = 'o')

fig, ax = pl.subplots(1,1)
im = ax.scatter(master_pd_duck[master_pd_nbn.acc > 0.5].RMM90, master_pd_nbn[master_pd_nbn.acc > 0.5].RMM90, c= master_pd_nbn['acc'], cmap = 'magma')
pl.colorbar(im)
pl.plot((0, 0.01),(0,0.01))

#
#
#
#
#




########################################plotting




# for trainsite in ['nbn_duck']:
#     for testsite in ['duck', 'nbn']:
#             mm = 0
#             modelname = 'resnet512_five_aug_{}'.format(mm)
#             vis_dir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/model_output/train_on_{}/{}/visualize/test_on_{}/'.format(trainsite, modelname, testsite)
#
#             if not load_variog:
#                 azi0, azi90 = return_variograms(trainsite, testsite, imgtype, x, y, vario_params_0, vario_params_90, modelname)
#                 variogram_pickle = {'azi0':azi0, 'azi90':azi90}
#
#                 with open(vis_dir + 'variogram_info_smallag.pickle', 'wb') as f:
#                     pickle.dump(variogram_pickle, f)
#
#
#             if load_variog:
#                 with open(vis_dir + 'variogram_info.pickle', 'rb') as f:
#                     variogram_pickle = pickle.load(f)
#                 azi0 = variogram_pickle['azi0']
#                 azi90 = variogram_pickle['azi90']
#
#
#             fig_state, ax_state = pl.subplots(5, 2, figsize = (8,8), tight_layout = True)
#             for si, state in enumerate(states):
#                 azi0_matrix = azi0[state]
#                 azi90_matrix = azi90[state]
#
#                 lag0 = azi0['lag0']
#                 lag90 = azi90['lag90']
#
#                 avg0 = np.mean(azi0_matrix, axis = 0)
#                 avg90 = np.mean(azi90_matrix, axis = 0)
#
#
#                 difference = avg0-1
#                 check = np.where(difference > 0)[0]
#                 if check.size>0:
#                     range0 = lag0[np.where(difference>0)[0][0]]
#                 else:
#                     range0 = lag0[np.where(difference == np.max(difference))[0][0]]
#
#                 difference = avg90-1
#                 check = np.where(difference > 0)[0]
#                 if check.size>0:
#                     range90 = lag90[np.where(difference>0)[0][0]]
#                 else:
#                     range90 = lag90[np.where(difference == np.max(difference))[0][0]]
#
#                 ax_state[si, 0].scatter(lag0[:-1], avg0[:-1])
#                 ax_state[si, 0].plot((0, ymax0), (1, 1), 'k')
#                 ax_state[si, 0].plot(lag0[:-1], avg0[:-1])
#                 ax_state[si, 0].set_ylim((0, 1.5))
#                 ax_state[si, 0].set_title('{0:0.2f}'.format(range0))
#
#                 ax_state[si,1].scatter(lag90[:-1], avg90[:-1])
#                 ax_state[si,1].plot(lag90[:-1], avg90[:-1])
#                 ax_state[si, 1].plot((0, ymax90), (1, 1), 'k')
#                 ax_state[si, 1].set_ylim((0, 1.5))
#                 ax_state[si, 1].set_title('{0:0.2f}'.format(range90))
#
#
#                 fig_state.suptitle('Average Variogram')
#
#                 num_cams = 0
#                 for ii in [1]:
#
#                     fig, ax = pl.subplots(5, 3, figsize = (8,6), tight_layout = True)
#                     modelname = 'resnet512_five_aug_0'
#
#                     vis_dir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/model_output/train_on_{}/{}/visualize/test_on_{}/'.format(trainsite, modelname, testsite)
#                     with open(vis_dir + 'BackCam_{}_{}_{}.pickle'.format(state, ii, imgtype), 'rb') as f:
#                         cams = pickle.load(f, encoding = 'latin1')
#
#                     for ci,cam in enumerate(cams):
#                         ax[ci, 0].pcolor(x, y, cam[::-1])
#
#                         ax[ci, 1].scatter(lag0, azi0_matrix[num_cams])
#                         ax[ci, 1].plot((0, ymax0), (1, 1), 'k')
#
#                         ax[ci, 2].scatter(lag90, azi90_matrix[num_cams])
#                         ax[ci, 2].plot((0, ymax90), (1, 1), 'k')
#                         ax[ci, 1].set_ylim(0, 1.5)
#                         ax[ci, 2].set_ylim(0, 1.5)
#                         num_cams += 1
#
#                     ax[0, 1].set_title('Azi: 0')
#                     ax[0,2].set_title('Azi: 90')
#                     ax[4, 1].set_xlabel('Cross-Shore')
#                     ax[4, 2].set_xlabel('Along Shore')
#
#                     fig.savefig(vis_dir + 'variogram_{}_{}_smallag.png'.format(state, ii))
#
#                 fig_state.savefig(vis_dir + 'variogram_AllStates_avg_testsitecompare_smallag.png')
