import seaborn as sns
import numpy as np
import matplotlib.pyplot as pl
import pickle
import geostatspy.GSLIB as GSLIB
import geostatspy.geostats as geostats
import pandas as pd
import time
from scipy.signal import argrelextrema, argrelmin

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



def argus_variograms(cam_df, vario_params):
    cam = cam_df['intensity']
    tmin = 0
    tmax = cam.max()
    isill = 1
    atol = 0
    bandh = 200


    lag_dist = vario_params['lag_dist']
    lag_tol = vario_params['lag_tol']
    nlag = int(vario_params['ymax']/lag_dist)
    azi = vario_params['azi']


    lag, gamma, npp = geostats.gamv(cam_df, 'x', 'y', 'intensity', tmin, tmax, lag_dist, lag_tol,nlag, azi, atol, bandh, isill=1)

    return lag, gamma, npp

def return_variograms(trainsite, testsite, imgtype, x, y, params_0_dict, params_90_dict, modelname):

        azi0 = {} #Return dictionaries with the variograms in arrays
        azi90 = {}

    
        for i, state in enumerate(states):
            azi0_matrix = np.zeros((5, 12)) #Matrices are size num_cams x lag distance
            azi90_matrix = np.zeros((5, 17))

            num_cams = 0 #index through the number of cams for each state, for each of the variogram matrices
             #corresponds to model number

            vis_dir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/model_output/train_on_{}/{}/visualize/test_on_{}/'.format(trainsite, modelname, testsite)


            for ii in [1]:

                with open(vis_dir + 'BackCam_{}_{}_{}.pickle'.format(state, ii, imgtype), 'rb') as f:
                    cams = pickle.load(f, encoding = 'latin1')
                for ci,cam in enumerate(cams):
                    cam_df = pd.DataFrame({'x':x.flatten(), 'y':y.flatten(), 'intensity':cam.flatten()}, index = range(len(x.flatten())))

                    lag0, gamma0, npp0 = argus_variograms(cam_df, params_0_dict)
                    azi0_matrix[num_cams] = gamma0

                    lag90, gamma90, npp90 = argus_variograms(cam_df, params_90_dict)
                    azi90_matrix[num_cams] = gamma90

                    num_cams += 1

            azi0.update({state:azi0_matrix})
            azi90.update({state:azi90_matrix})
            azi0.update({'lag0':lag0})
            azi90.update({'lag90':lag90})

        return azi0, azi90

def img_variograms(x, y, testsite, params_dict):

    from PIL import Image
    imgdir = {'nbn': '/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/',
            'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/match_nbn/'}

    with open('../labels/{}_daytimex_valfiles.final.pickle'.format(testsite), 'rb') as f:
        test_IDs = pickle.load(f)

    img_variog = {}

    for id in test_IDs:
        val_img = imgdir[testsite] + id
        val_img = Image.open(val_img)
        val_img = np.array(val_img.resize((128, 128)))

        val_df = pd.DataFrame({'x':x.flatten(), 'y':y.flatten(), 'intensity':val_img.flatten()})

        lag, gamma, npp = argus_variograms(val_df, params_dict)
        gamma = np.expand_dims(gamma, axis = 0)

        img_variog.update({id:gamma})


    img_variog['lag'] = lag

    return img_variog

def variogram_chars(variog, lag):

    h = lag[1] - lag[0]
    idx_max = argrelextrema(variog, np.greater, order = 2)[0]
    idx_min = argrelmin(variog, order = 2)[0]
    if len(idx_max) == 0:
        idx_max = np.where(variog == np.max(variog))[0]
        idx_min = np.where(variog == np.min(variog))[0]
    
    idx_max_half = int(idx_max[0]/2)

    if len(idx_max) < 2:
        idx_max = np.concatenate((idx_max, idx_max))

    v0 = variog[1]
    v1 = variog[2]
    v2 = variog[3]
    v3 = variog[4]
    l0 = lag[1]
        
    RVF = 1/v0 # relationship between spatial correlation at long and shore distances. High with high variability at long distances and low variability at short distances
    RSF = v1/v0 #changes in variability of data at short distances
    FDO = (v1 - v0)/h#variability in changes of data at short distances
    SDT = (v3 - 2*v2 + v1)/(h**2) #concavity or convexity (val>0: concave, homogenous at short distances. val<0:convex, heterogenous at short distances
    FML = variog[idx_max[0]] #granularity of the image
    MFM = np.mean(variog[:idx_max[0]]) #changes in variability of the data, related to concavity or convexity of semivariogram
    VFM = np.var(variog[:idx_max[0]])
    DMF = MFM - v0 #decreasing rate of spatial correlation
    RMM = variog[idx_max[0]]/MFM
    SDF = variog[idx_max[0]] - 2*variog[idx_max_half] + variog[0]
    AFM = h/2*(v0 + np.sum(variog[1:idx_max[0]-1]) + variog[idx_max[0]]) - (v0*(lag[idx_max[0]] - l0)) #semivariogram curvature info
    
    DMS = lag[idx_max[1]] - lag[idx_max[0]] #size of regularity of structural pattern of a texture in an image
    try:
        DMM = lag[idx_min[0]] - lag[idx_max[0]]
    except:
        DMM = np.nan
    HA = ((h/2) * (variog[idx_max[0]] + 2 *np.sum(variog[idx_max[0]+1:idx_max[1]-1]) + variog[idx_max[1]]))/(0.5*(lag[idx_max[1]] - lag[idx_max[0]])*(variog[idx_max[1]]+variog[idx_max[0]]))

    return RVF, RSF, FDO, SDT, FML, MFM, VFM, DMF, RMM, SDF, AFM, DMS, DMM, HA

def produce_omnivariograms():
    for testsite in ['nbn', 'duck']:
        x,y = np.meshgrid(range(128), range(128))
        ymax = 64
        load_variog = 0 #boolean to load finished calculations or to generate new variograms
        for ai, azi in enumerate([0, 30, 60, 90, 120, 150]):
            vario_params = {'lag_dist':3, 'lag_tol':1, 'ymax':ymax, 'azi':azi}
            img_variog = img_variograms(x, y, testsite, vario_params)

            if ai == 0:
                full_variog = img_variog.copy()

            if ai>0:
                for pid in img_variog.keys():
                    if 'lag' not in pid:
                        full_variog[pid] = np.concatenate((full_variog[pid], img_variog[pid]), axis = 0)


        mean_variog = {}
        for pid in full_variog.keys():
            mean_v = np.mean(full_variog[pid], axis = 0)
            mean_variog.update({pid:mean_v})

        variogram_pickle = {'full_variog':full_variog, 'mean_variog':mean_variog}
        with open('{}_variograms_omnidir.pickle'.format(testsite), 'wb') as f:
            pickle.dump(variogram_pickle, f)


def produce_cross_shore_intensity():
    for testsite in ['nbn', 'duck']:
        from PIL import Image
        imgdir = {'nbn': '/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/',
                'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/match_nbn/'}

        with open('../labels/{}_daytimex_valfiles.final.pickle'.format(testsite), 'rb') as f:
            test_IDs = pickle.load(f)

        img_intensity = {}

        for id in test_IDs:
            val_img = imgdir[testsite] + id
            val_img = Image.open(val_img)
            val_img = np.array(val_img.resize((512, 512)))

            val_img_x = np.mean(val_img, axis = 1)
            val_img_x_norm = val_img_x/val_img_x.max()

            img_intensity.update({id:val_img_x_norm})

        with open('img_intensity_{}.pickle'.format(testsite), 'wb') as f:
            pickle.dump(img_intensity, f)


def intensity_CDF(img_intensity):

    intensity_cdf = {}

    for pid, val_img_x_norm in img_intensity.items():
        img_cdf = np.zeros((len(val_img_x_norm)))

        for vi in range(len(val_img_x_norm)):

            img_cdf[vi] = val_img_x_norm[:vi]/val_img_x_norm.sum()

    intensity_cdf.update({pid: img_cdf})

    return intensity_cdf



produce_cross_shore_intensity()
master_pd_duck = img_votes('nbn_duck', 'duck', 'resnet512_five_aug')
master_pd_nbn = img_votes('nbn_duck', 'nbn', 'resnet512_five_aug')
master_pd_duck = img_accuracies(master_pd_duck)
master_pd_nbn = img_accuracies(master_pd_nbn)
master_pd_duck['testsite'] = 'duck'
master_pd_nbn['testsite'] = 'nbn'
full_img_pd = pd.concat((master_pd_duck, master_pd_nbn))

imgdirs = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/match_nbn/',
           'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}
states = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
imgtype = 'resize_128'
maxindsdist = {}
for testsite in ['duck', 'nbn']:

    with open('img_intensity_{}.pickle'.format(testsite), 'rb') as f:
        img_pickle = pickle.load(f)

    for statenum in range(len(states)):
        state_pids = full_img_pd[(full_img_pd.testsite == testsite) & (full_img_pd.truth == statenum)].index
        for i in [1,5,10,15]:
            #fig, ax = pl.subplots(5, 2, figsize = (15,15))
            for j in range(5):
                xd = []
                pid = state_pids[i+j]
                I = img_pickle[pid]
                img = pl.imread(imgdirs[testsite]+pid)
                maxinds_pick = []
                if statenum > 2:
                    maxinds = argrelextrema(I, np.greater, order = 3)[0]
                    maxinds_pick = np.array([mm for mm in maxinds if I[mm]>0.4])

                    if len(maxinds_pick) > 2:
                        sortedI = np.argsort(I[maxinds_pick])
                        maxinds_pick = maxinds_pick[sortedI[-2:]]
                        print(maxinds_pick)

                if statenum <= 2:
                    maxI = np.where(I == np.max(I))

                # ax[j, 0].imshow(img)
                # ax[j, 0].set_title(pid.split('.')[0])
                # ax[j, 1].scatter(range(512), I)


                # if len(maxinds_pick) > 0 and statenum > 2:
                #     ax[j, 1].scatter(maxinds_pick, I[maxinds_pick], color = 'y')
                #
                if statenum <= 2:
                    # ax[j,1].plot(maxI, I[maxI], 'yo')
                    maxinds_pick = maxI

                if len(maxinds_pick) == 2:
                    xd = maxinds_pick[1] - maxinds_pick[0]

                if len(maxinds_pick) < 2:
                    xd = maxI

                maxindsdist.update({pid: xd})


            # fig.suptitle(testsite + ' Image I State {}'.format(states[statenum]))
            # fig.savefig('/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/visualizations/plots/I_wpeaks{}_{}_{}.png'.format(testsite, states[statenum], i))


for pid in full_img_pd.index:
    full_img_pd.loc[pid, 'xd'] = maxindsdist[pid][0]





#
#
#
# for mi, master_pd in enumerate([master_pd_nbn, master_pd_duck]):
#     if mi == 0:
#         testsite = 'nbn'
#     if mi == 1:
#         testsite = 'duck'
#     with open('{}_variograms_omnidir.pickle'.format(testsite), 'rb') as f:
#         variogram_pickle = pickle.load(f)
#     img_variog0 = variogram_pickle['mean_variog']
#     img_variog90 = variogram_pickle['img_variog90']
#     lag0 = img_variog0['lag0']
#     lag90 = img_variog90['lag90']
#
#     for vv, var_dict in enumerate([img_variog0, img_variog90]):
#         if vv == 0:
#             lag = lag0
#             az = 0
#         if vv == 1:
#             lag = lag90
#             az = 90
#         RVF_list = []
#         RSF_list = []
#         FDO_list = []
#         SDT_list = []
#         FML_list = []
#         MFM_list = []
#         VFM_list = []
#         DMF_list = []
#         RMM_list = []
#         SDF_list = []
#         AFM_list = []
#         DMS_list = []
#         DMM_list = []
#         HA_list = []
#         for pid in master_pd.index.values:
#             variog = var_dict[pid]
#             RVF, RSF, FDO, SDT, FML, MFM, VFM, DMF, RMM, SDF, AFM, DMS, DMM, HA = variogram_chars(variog[1:], lag[1:])
#             RVF_list.append(RVF)
#             RSF_list.append(RSF)
#             FDO_list.append(FDO)
#             SDT_list.append(SDT)
#             FML_list.append(FML)
#             MFM_list.append(MFM)
#             VFM_list.append(VFM)
#             DMF_list.append(DMF)
#             RMM_list.append(RMM)
#             SDF_list.append(SDF)
#             AFM_list.append(AFM)
#             DMS_list.append(DMS)
#             DMM_list.append(DMM)
#             HA_list.append(HA)
#
#         master_pd['RVF{}'.format(az)] = RVF_list
#         master_pd['RSF{}'.format(az)] = RSF_list
#         master_pd['FDO{}'.format(az)] = FDO_list
#         master_pd['SDT{}'.format(az)] = SDT_list
#         master_pd['FML{}'.format(az)] = FML_list
#         master_pd['MFM{}'.format(az)] = MFM_list
#         master_pd['VFM{}'.format(az)] = VFM_list
#         master_pd['DMF{}'.format(az)] = DMF_list
#         master_pd['RMM{}'.format(az)] = RMM_list
#         master_pd['SDF{}'.format(az)] = SDF_list
#         master_pd['SDF{}'.format(az)] = SDF_list
#         master_pd['AFM{}'.format(az)] = AFM_list
#         master_pd['DMS{}'.format(az)] = DMS_list
#         master_pd['DMM{}'.format(az)] = DMM_list
#         master_pd['HA{}'.format(az)] = HA_list
#
#
# vis_dir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/visualizations'
#
#
#
#
#
# metrics = [name for name in master_pd.columns if '0' in name and 'preds' not in name and '90' not in name]
# metrics = [name for name in master_pd.columns if '90' in name]
# fig, axes = pl.subplots(7,2, tight_layout= True)
# for mi, metric in enumerate(metrics):
#     ax = axes.ravel('F')[mi]
# for metric in metrics:
#     fig, axes = pl.subplots(5,1)
#     for statenum in range(5):
#         ax = axes[statenum]
#         sns.scatterplot(x = metric, y = 'acc', data = master_pd_duck[master_pd_duck.truth == statenum], ax = ax, legend = False, color = 'blue')
#         sns.scatterplot(x = metric, y = 'acc', data = master_pd_nbn[master_pd_nbn.truth == statenum], ax = ax, color = 'red')
#
# fig, ax = pl.subplots(1,1)
# sns.scatterplot(x = 'RMM90', y = 'acc', data = master_pd_duck, hue = 'truth', marker = 'x')
# sns.scatterplot(x = 'RMM90', y = 'acc', data = master_pd_nbn, hue = 'truth', marker = 'o')
#
# fig, ax = pl.subplots(1,1)
# im = ax.scatter(master_pd_duck[master_pd_nbn.acc > 0.5].RMM90, master_pd_nbn[master_pd_nbn.acc > 0.5].RMM90, c= master_pd_nbn['acc'], cmap = 'magma')
# pl.colorbar(im)
# pl.plot((0, 0.01),(0,0.01))
#
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
