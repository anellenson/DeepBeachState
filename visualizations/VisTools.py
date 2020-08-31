import numpy as np
import pandas as pd
import pickle
import geostatspy.GSLIB as GSLIB
import geostatspy.geostats as geostats
from PIL import Image
from scipy.signal import argrelextrema, argrelmin


class Variograms():

    def load_variograms(self):

        for ti, testsite in enumerate(['nbn', 'duck']):
            with open('{}_variograms_fulllag.pickle'.format(testsite), 'rb') as f:
                variogram_pickle = pickle.load(f)
            if ti == 0:
                img_variog0 = variogram_pickle['img_variog0']
                img_variog90 = variogram_pickle['img_variog90']
                lag0 = variogram_pickle['lag0']
                lag90 = variogram_pickle['lag90']
            if ti > 0:
                img_variog0.update(variogram_pickle['img_variog0'])
                img_variog90.update(variogram_pickle['img_variog90'])

        return img_variog0, img_variog90, lag0, lag90

    def variogram_chars_df(self, img_variog0, img_variog90, lag0, lag90, master_pd):

        for vv, var_dict in enumerate([img_variog0, img_variog90]):
            if vv == 0:
                lag = lag0
                az = 0
            if vv == 1:
                lag = lag90
                az = 90
            RVF_list = []
            RSF_list = []
            FDO_list = []
            SDT_list = []
            FML_list = []
            MFM_list = []
            VFM_list = []
            DMF_list = []
            RMM_list = []
            SDF_list = []
            AFM_list = []
            DMS_list = []
            DMM_list = []
            HA_list = []
            for pid in master_pd.index.values:
                variog = var_dict[pid].squeeze()
                RVF, RSF, FDO, SDT, FML, MFM, VFM, DMF, RMM, SDF, AFM, DMS, DMM, HA = self.variogram_chars(variog[1:], lag[1:])
                RVF_list.append(RVF)
                RSF_list.append(RSF)
                FDO_list.append(FDO)
                SDT_list.append(SDT)
                FML_list.append(FML)
                MFM_list.append(MFM)
                VFM_list.append(VFM)
                DMF_list.append(DMF)
                RMM_list.append(RMM)
                SDF_list.append(SDF)
                AFM_list.append(AFM)
                DMS_list.append(DMS)
                DMM_list.append(DMM)
                HA_list.append(HA)

            master_pd['RVF{}'.format(az)] = RVF_list
            master_pd['RSF{}'.format(az)] = RSF_list
            master_pd['FDO{}'.format(az)] = FDO_list
            master_pd['SDT{}'.format(az)] = SDT_list
            master_pd['FML{}'.format(az)] = FML_list
            master_pd['MFM{}'.format(az)] = MFM_list
            master_pd['VFM{}'.format(az)] = VFM_list
            master_pd['DMF{}'.format(az)] = DMF_list
            master_pd['RMM{}'.format(az)] = RMM_list
            master_pd['SDF{}'.format(az)] = SDF_list
            master_pd['SDF{}'.format(az)] = SDF_list
            master_pd['AFM{}'.format(az)] = AFM_list
            master_pd['DMS{}'.format(az)] = DMS_list
            master_pd['DMM{}'.format(az)] = DMM_list
            master_pd['HA{}'.format(az)] = HA_list

        return master_pd

    def average_variograms(self, statenum, full_img_pd, testsite, img_variog0, img_variog90):
            columns = [] #columns correspond to images
            for pi, pid in enumerate(full_img_pd[(full_img_pd.truth == statenum) & (full_img_pd.testsite == testsite)].index):
                if pi == 0:
                    variog0 = img_variog0[pid]
                    variog90 = img_variog90[pid]
                    variog0 = np.expand_dims(variog0, axis = 1)
                    variog90 = np.expand_dims(variog90, axis = 1)
                    columns.append('varpt{}'.format(pi))
                if pi>0:
                    v0 = np.expand_dims(img_variog0[pid], axis = 1)
                    v90 = np.expand_dims(img_variog90[pid], axis = 1)
                    variog0 = np.concatenate((variog0,v0), axis = 1)
                    variog90 = np.concatenate((variog90, v90), axis = 1)
                    columns.append('varpt{}'.format(pi))

            variog0_df = pd.DataFrame(columns = columns, data = variog0.squeeze().T)
            variog90_df = pd.DataFrame(columns = columns, data = variog90.squeeze().T)

            return variog0_df, variog90_df

    def variogram_chars(self, variog, lag):

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

    def produce_variograms(self):
        for testsite in ['duck']:
            x,y = np.meshgrid(range(128), range(128))
            ymax = 128
            load_variog = 0 #boolean to load finished calculations or to generate new variogram
            img_variog0 = {}
            img_variog90 = {}# s
            for ai, azi in enumerate([0, 90]):
                vario_params = {'lag_dist':3, 'lag_tol':1, 'ymax':ymax, 'azi':azi}
                img_variog, lag = self.img_variograms(x, y, testsite, vario_params)

                if azi == 0:
                    lag0 = lag.copy()

                if azi == 90:
                    lag90 = lag.copy()

                for pid in img_variog.keys():
                    if 'lag' not in pid:
                        if azi == 0:
                            img_variog0[pid] = img_variog[pid]

                    if azi == 90:
                        img_variog90[pid] = img_variog[pid]

            variogram_pickle = {'img_variog0':img_variog0, 'img_variog90':img_variog90, 'lag0':lag0, 'lag90':lag90}
            with open('{}_variograms_fulllag.pickle'.format(testsite), 'wb') as f:
                pickle.dump(variogram_pickle, f)


    def img_variograms(self, x, y, testsite, vario_params):

        from PIL import Image
        imgdir = {'nbn': '/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/',
                'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/'}

        with open('../labels/{}_daytimex_valfiles.final.pickle'.format(testsite), 'rb') as f:
            test_IDs = pickle.load(f)

        img_variog = {}

        tmin = 0
        atol = 0
        bandh = 200


        lag_dist = vario_params['lag_dist']
        lag_tol = vario_params['lag_tol']
        nlag = int(vario_params['ymax']/lag_dist)
        azi = vario_params['azi']


        for id in test_IDs:
            val_img = imgdir[testsite] + id
            val_img = Image.open(val_img)
            val_img = np.array(val_img.resize((128, 128)))

            val_df = pd.DataFrame({'x':x.flatten(), 'y':y.flatten(), 'intensity':val_img.flatten()})
            cam = val_df['intensity']
            tmax = cam.max()

            lag, gamma, npp = geostats.gamv(val_df, 'x', 'y', 'intensity', tmin, tmax, lag_dist, lag_tol,nlag, azi, atol, bandh, isill=1)
            gamma = np.expand_dims(gamma, axis = 0)

            img_variog.update({id:gamma})


        return img_variog, lag

    def return_variograms(self, trainsite, testsite, imgtype, x, y, params_0_dict, params_90_dict, modelname):

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

                        lag0, gamma0, npp0 = self.argus_variograms(cam_df, params_0_dict)
                        azi0_matrix[num_cams] = gamma0

                        lag90, gamma90, npp90 = self.argus_variograms(cam_df, params_90_dict)
                        azi90_matrix[num_cams] = gamma90

                        num_cams += 1

                azi0.update({state:azi0_matrix})
                azi90.update({state:azi90_matrix})
                azi0.update({'lag0':lag0})
                azi90.update({'lag90':lag90})

            return azi0, azi90

class Intensity():

    def __init__(self, testsites, imgdirs, states):
        self.testsites = testsites
        self.imgdirs = imgdirs
        self.states = states

    def intensity_CDF(self, img_intensity):

        intensity_cdf = {}

        for pid, val_img_x_norm in img_intensity.items():
            img_cdf = np.zeros((len(val_img_x_norm)))

            for vi in range(len(val_img_x_norm)):

                img_cdf[vi] = val_img_x_norm[:vi]/val_img_x_norm.sum()

        intensity_cdf.update({pid: img_cdf})

        return intensity_cdf

    def return_transects_pd(self, full_img_pd):
        transect_dict = {}
        for statenum in range(len(self.states)):
            transect_list = []

            for testsite in ['nbn', 'duck']:

                state_pids = full_img_pd[(full_img_pd.testsite == testsite) & (full_img_pd.truth == statenum)].index

                for pi, pid in enumerate(state_pids):
                    img = Image.open(self.imgdirs[testsite] + pid)
                    img = img.resize((512,512))
                    img_array = np.array(img)
                    transect = img_array[:400,256]
                    transect_list += list(transect)

            transect_dict.update({self.states[statenum]:transect_list})

        transect_pd = pd.DataFrame(data = transect_dict)

        testsite_list = ['nbn']*int(len(transect_list)/2) + ['duck']*int(len(transect_list)/2)
        pts = list(range(400)) * 50
        transect_pd['testsite'] = testsite_list
        transect_pd['pixelpts'] = pts

        return transect_pd


    def find_xs_points(self, full_img_pd):
        xsinds = {}
        fig, ax = pl.subplots(1,2)
        fig.set_size_inches([15,15])

        for testsite in ['duck', 'nbn']:

            for statenum in [3,4]:
                state_pids = full_img_pd[(full_img_pd.testsite == testsite) & (full_img_pd.truth == statenum)].index

                for pid in state_pids:
                    if pid not in xsinds.keys():
                        img = Image.open(imgdirs[testsite] + pid)
                        img = img.resize((512,512))
                        img_array = np.array(img)
                        transect = img_array[:,256]
                        ax[0].imshow(img)
                        ax[0].plot((256, 256),(0,512), 'k')


                        ax[1].plot(range(512),transect)
                        ax[1].set_ylim((0, 255))
                        ax[1].set_title(full_img_pd.loc[pid]['acc'])
                        points = pl.ginput(n =2)
                        pts = [int(points[0][1]), int(points[1][0])]
                        ax[1].scatter(pts, transect[pts])
                        pl.waitforbuttonpress()
                        xsinds.update({pid:pts})
                        ax[1].clear()

                        with open('xs_pts.pickle', 'wb') as f:
                            pickle.dump(xsinds, f)

            return full_img_pd





