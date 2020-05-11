import numpy as np
import matplotlib.pyplot as pl
import pickle
import geostatspy.GSLIB as GSLIB
import geostatspy.geostats as geostats
import pandas as pd



x,y = np.meshgrid(range(512), range(512))
x = 900/512*x
y = 300/512*y
states = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
color = {'nbn':'red', 'duck':'darkblue', 'nbn_duck':'purple'}
manuscript_plot_dir = '/home/aquilla/aellenso/Research/DeepBeach/resnet_manuscript/plots/'
save_for_manuscript = False
for testsite in ['nbn', 'duck']:
    for trainsite in ['nbn_duck']:

        for i, state in enumerate(states):

            mm = 0
            modelname = 'resnet512_five_aug_{}'.format(mm)
            vis_dir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/model_output/train_on_{}/{}/visualize/test_on_{}/'.format(trainsite, modelname, testsite)

            for ii in [1]:
                fig, ax = pl.subplots(5,3, figsize = [15,15])
                with open(vis_dir + 'BackCam_{}_{}.pickle'.format(state, ii), 'rb') as f:
                    cams = pickle.load(f, encoding = 'latin1')
                for ci, cam in enumerate(cams):
                    im = ax[ci,0].pcolor(cam[::-1])
                    pl.colorbar(im, ax = ax[ci,0])

                    cam_df = pd.DataFrame({'x':x.flatten()[::10], 'y':y.flatten()[::10], 'intensity':cam.flatten()[::10]}, index = range(len(x.flatten()[::10])))
                    tmin = 0
                    tmax = cam.max()
                    nxlag = 50
                    nylag = 20
                    dxlag = 30
                    dylag = 15
                    minnp = 2
                    isill = 0
                    lag_dist = 10
                    lag_tol = 1
                    nlag = 50
                    bandh = 200
                    azi =[0,90]
                    atol = 22.5

                    #varmap, numpairs = geostats.varmapv(cam_df, 'x', 'y', 'intensity', tmin = tmin, tmax = tmax, nxlag = nxlag, nylag = nylag, dxlag = dxlag, dylag = dylag, minnp = minnp, isill=isill)
                    lag, gamma, npp = geostats.gamv(cam_df, 'x', 'y', 'intensity', tmin, tmax, lag_dist, lag_tol,nlag, azi[0], atol, bandh, isill=0)
                    ax[ci, 1].scatter(lag, gamma)
                    ax[ci, 1].set_title('Azimuth 0')

                    lag, gamma, npp = geostats.gamv(cam_df, 'x', 'y', 'intensity', tmin, tmax, lag_dist, lag_tol,nlag, azi[1], atol, bandh, isill=0)
                    ax[ci, 2].scatter(lag, gamma)
                    ax[ci, 2].set_title('Azimuth 90')

                fig.savefig(vis_dir, 'variogram_{}_{}_every10thpixel.png'.format(state, ii))
                print('State {} Printed'.format(state))

