import matplotlib.pyplot as pl
import pickle
from PIL import Image


def fig_2_beachstates(plotfolder):
    statenum_dict = {'nbn': {'Ref':'1436477406', 'LTT':'1358370027', 'TBR':'1382648427', 'RBB':'1331586028', 'LBT':'1449451807'},'duck': {'Ref':'1330534800', 'LTT':'1343664000', 'TBR':'1400515200', 'RBB':'1394816400', 'LBT':'1328634000'}}

    imgdir = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/match_nbn/', 'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}

    for trainsite in ['duck', 'nbn']:

        statenum = statenum_dict[trainsite]
        with open('labels/{}_daytimex_valfiles.no_aug.pickle'.format(trainsite), 'rb') as f:
            valfiles = pickle.load(f)

        classes = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
        full_classnames = ['Reflective (Ref)', 'Low Tide Terrace (LTT)', 'Transverse Bar Rip (TBR)', 'Rhythmic Bar Beach (RBB)', 'Longshore Bar Trough (LBT)']
        test_IDs = []
        for beachstate in classes:
            class_pids = [aa for aa in valfiles if statenum[beachstate] in aa]
            test_IDs += class_pids

        test_IDs = [imgdir[trainsite] + '/'+tt for tt in test_IDs]

        fig, ax = pl.subplots(5,1, tight_layout = {'rect':[0,0, 1,0.98]}, figsize = [4,9])
        for ii in range(len(classes)):
            img = Image.open(test_IDs[ii])
            ax[ii].imshow(img, cmap = 'gray')
            ax[ii].axis('off')
            ax[ii].set_title(full_classnames[ii], fontsize = 12)

        if trainsite == 'duck':
            pl.suptitle('Duck Beach States'.format(trainsite))
            pl.savefig(plotfolder+'fig2_duck.png')

        if trainsite == 'nbn':
            pl.suptitle('Narrabeen Beach States'.format(trainsite), fontweight = 'bold')
            pl.savefig(plotfolder+'fig2_nbn.png')


plotfolder = '/home/aquilla/aellenso/Research/DeepBeach/resnet_manuscript/plots/'
fig_2_beachstates(plotfolder)
