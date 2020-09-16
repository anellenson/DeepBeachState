import matplotlib.pyplot as pl
import pickle
from PIL import Image
import matplotlib.pyplot as pl
import plotTools as pt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics


def fig_2_beachstates(plotfolder):
    statenum_dict = {'nbn': {'Ref':'1436477406', 'LTT':'1358370027', 'TBR':'1382648427', 'RBB':'1331586028', 'LBT':'1449451807'},'duck': {'Ref':'1330534800', 'LTT':'1343664000', 'TBR':'1400515200', 'RBB':'1394816400', 'LBT':'1328634000'}}

    imgdir = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/full/', 'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}

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


def fig_5_transfer(plotfolder):

    all_results_df = pd.DataFrame(columns = ['train_site', 'test_site','corr-coeff', 'f1', 'nmi', 'model_type'])
    modelnames = []
    for ti, trainsite in enumerate(['nbn', 'duck', 'nbn_duck']):
        if trainsite == 'nbn' or trainsite == 'duck':
            modelnames = ['resnet512_five_aug_trainloss_']
        if trainsite == 'nbn_duck':
            for addsite in ['nbn', 'duck']:
                for percentage in [5, 10, 15, 25]:
                    modelname = 'resnet512_five_aug_minsite{}_{}imgs_'.format(addsite, percentage)
                    modelnames.append(modelname)


        #modelnames = ['resnet512_five_aug_']
        ########
        plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
        out_folder = 'model_output/train_on_{}/'.format(trainsite)

        skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
        if ti == 0:
            all_results_df = skc.gen_skill_df(ensemble = True)
            if trainsite == 'nbn':
                all_results_df.loc[all_results_df.model_type == 'resnet512_five_aug_trainloss_', 'model_type'] = 'resnet512_nbn'
        if ti >0:
            all_results_df = pd.concat((all_results_df, skc.gen_skill_df(ensemble=True)))
            if trainsite == 'duck':
                all_results_df.loc[all_results_df.model_type == 'resnet512_five_aug_trainloss_', 'model_type'] = 'resnet512_duck'


    relabel_inds = {'resnet512_nbn':0, 'resnet512_five_aug_minsiteduck_5imgs_':1, 'resnet512_five_aug_minsiteduck_10imgs__':2,
                    'resnet512_five_aug_minsiteduck_15imgs_':3, 'resnet512_five_aug_minsiteduck_25imgs_':4, 'resnet512_five_aug_trainloss_':5, 'resnet512_five_aug_minsitenbn_25imgs_':6, 'resnet512_five_aug_minsitenbn_15imgs_':7,
                    'resnet512_five_aug_minsitenbn_10imgs_':8,'resnet512_five_aug_minsitenbn_5imgs_':9,'resnet512_duck':10}

    labels = ['100 \n0', '95 \n5', '90 \n10', '85 \n15', '75 \n25', '50 \n50', '25 \n75', '15 \n85', '10 \n90',
              '5 \n 95', '0 \n100']

    for modelname in relabel_inds.keys():
        all_results_df.loc[all_results_df.model_type == modelname, 'num'] = relabel_inds[modelname]

    palette = {'both':'black', 'nbn':'red', 'duck':'blue'}
    fig, ax = pl.subplots(1,1, sharex = True, tight_layout = {'rect':[0, 0, 1, 0.95]}, figsize = [8, 6])
    fig.set_size_inches(5, 2.5)
    a = sns.barplot(x = 'num', y ='f1', hue = 'test_site', data = all_results_df, ax = ax, palette = palette)
    a.set_xticks(range(13))
    a.set_xticklabels(labels)
    a.legend_.remove()
    a.set_xlabel('')
    ax.set_ylabel('F-Score')
    a.text(-1, -0.15, 'Nbn', color = 'red')
    a.text(-1, -0.25, 'Duck', color = 'blue')
    pl.savefig(plotfolder + 'fig5_results_bar.png')

    fig, ax = pl.subplots(1,1,sharex = True, tight_layout = {'rect':[0, 0, 1, 0.97]}, figsize = [5,5])
    a = sns.lineplot(x = 'num', y = 'f1', hue = 'test_site', data = all_results_df, ax = ax, palette = palette)
    ax.set_ylim((0,1))
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels)
    a.text(-1.2, -0.05, 'Nbn', color = 'red')
    a.text(-1.3, -0.09, 'Duck', color = 'blue')
    pl.grid()
    leg = a.get_legend()
    a.legend(ncol =1, loc='lower right')
    leg = a.get_legend()
    a.legend_.set_title('Test Loc')
    for t, l in zip(leg.texts, ['', 'Duck', 'Nbn', 'Combined']): t.set_text(l)
    ax.set_ylabel('F-Score')
    ax.set_xlabel('Percentage of Images from Each Site')
    pl.savefig(plotfolder + 'fig5_results_line.png')

def fig_4_results(plotfolder):

    modelnames = ['resnet512_five_aug_trainloss_']
    testsites = ['nbn', 'duck']
    for ti, trainsite in enumerate(['nbn', 'duck', 'nbn_duck']):
        plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
        out_folder = 'model_output/train_on_{}/'.format(trainsite)
        skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
        if ti == 0:
            all_results_df = skc.gen_skill_df(testsites = testsites, ensemble = True)
        if ti >0:
            all_results_df = pd.concat((all_results_df, skc.gen_skill_df(testsites = testsites, ensemble=True)))

    # modelnames = ['resnet512_five_aug_trainloss_']
    # trainsite = 'nbn_duck'
    # skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
    # all_results_df = pd.concat((all_results_df, skc.gen_skill_df(testsites = testsites, ensemble=True)))
    # all_results_df.loc[all_results_df.model_type == 'resnet512_five_aug_trainloss_', 'train_site'] = 'nbn_duck_full'

    sns.set_color_codes('bright')
    fig, ax = pl.subplots(1,1, sharex = True, tight_layout = {'rect':[0, 0, 0.93, 0.95]})
    fig.set_size_inches(4, 4)
    metric = 'f1'
    a = sns.boxplot(x = 'train_site', y =metric, hue = 'test_site', data = all_results_df, ax = ax, palette = {'both':'black', 'nbn':'salmon', 'duck':'blue'})
    a.plot((-0.5, 2.5), (0.57, 0.57), '--', color = 'salmon')
    a.plot((-0.5, 2.5), (0.79, 0.79), '--', color = 'blue')
    a.text(2.55, 0.79, 'Duck', color = 'blue')
    a.text(2.55, 0.57, 'Nbn', color = 'salmon')
    a.set_xlim((-0.5, 2.5))
    #a = sns.barplot(x = metric, y ='train_site', hue = 'test_site', data = all_results_df[all_results_df.model_type == model], ax = ax[mi], palette = {'b', 'salmon'}, )
    leg = a.get_legend()
    a.legend(ncol =3, loc='lower right')
    leg = a.get_legend()
    a.legend_.set_title('Test Site')
    for t, l in zip(leg.texts, ['Nbn', 'Duck']): t.set_text(l)

    a.grid()
    ax.set_ylabel('')
    ax.set_ylim((0, 1))
    pl.suptitle('F-Score')
    ax.set_xticklabels(['Nbn', 'Duck', 'Combined', 'Combined-100'])
    ax.set_xlabel('Train Site')
    pl.savefig(plotfolder + 'fig4_results_boxplot.png')


def fig_6_conftables(plotfolder):
    test_type = 'val' #'transfer' or 'val', tells the function which file to pull
    testsites = [['nbn', 'duck'], ['nbn','duck'], ['nbn', 'duck']]
    modelnames = ['resnet512_five_aug_trainloss_']
    #The ensemble switch will show the average (True) or the best performing model (False)
    for ti, trainsite in enumerate(['nbn', 'duck','nbn_duck']):
        out_folder = 'model_output/train_on_{}/'.format(trainsite)
        skc = pt.skillComp(modelnames, plotfolder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
        plot_fname = 'fig6_conftable_{}_100images.png'.format(trainsite)
        _, fig, axes = skc.gen_conf_matrix(test_type, testsites[ti], plot_fname, average = True)
        if trainsite == 'nbn_duck':
            fig.suptitle('Combined CNN')
        if trainsite == 'nbn':
            fig.suptitle('Nbn CNN')
        if trainsite == 'duck':
            fig.suptitle('Duck CNN')
        axes[0].set_title('Test at Nbn')
        axes[1].set_title('Test at Duck')
        fig.savefig(plotfolder + plot_fname)



def fig_3_auth_conftables(plotfolder):
    states = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
    fig, ax = pl.subplots(2,1, tight_layout = {'rect':[0,0,0.95, 0.95]}, sharex = True)
    fig.set_size_inches(4, 5)
    for ti, testsite in enumerate(['nbn', 'duck']):
        f1_list = []
        for pi,person_name in enumerate(["GW", "KS", "KS1", 'JS']):
            with open('labels/{}_daytimex_valfiles_allauthors_df.pickle'.format(testsite), 'rb') as f:
                vallabels_df = pickle.load(f)
            labels_1 = vallabels_df['AE']
            labels_2 = vallabels_df[person_name]

            labels_2 = labels_2[labels_1.notna()]
            labels_1 = labels_1[labels_1.notna()]

            vallabels1 = [vv for vv in labels_1[labels_2.notna()].values]
            vallabels2 = [vv for vv in labels_2[labels_2.notna()].values]

            if not vallabels2  == []:
                if pi == 0:
                    conf_matrix = metrics.confusion_matrix(vallabels1, vallabels2)
                    conf_matrix = np.expand_dims(conf_matrix, axis = 0)
                if pi>0:
                    conf_matrix_1 = metrics.confusion_matrix(vallabels1, vallabels2)
                    conf_matrix_1 = np.expand_dims(conf_matrix_1, axis = 0)
                    conf_matrix = np.concatenate((conf_matrix, conf_matrix_1), axis = 0)

                f1 = metrics.f1_score(vallabels1, vallabels2, average = 'weighted')
                f1_list.append(f1)


        sk = pt.skillComp([], '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/prep_files/plots/', '/home/', testsite)

        sk.confusionTable(conf_matrix, states, fig, ax[ti], testsite, testsite + ' F1: {0:0.2f}'.format(f1, np.std(f1_list)), ensemble = True)
        ax[0].set_title('Narrabeen')
        ax[1].set_title('Duck')
        ax[0].set_ylabel('Truth')
        ax[1].set_ylabel('Truth')
        ax[1].set_xlabel('Co-Authors')
        ax[0].set_xlabel('')

        fig.suptitle('Human Agreement')
        fig.savefig(plotfolder + 'fig3_conftable_coauthors.png')

plotfolder = '/home/aquilla/aellenso/Research/DeepBeach/resnet_manuscript/plots/'
#fig_2_beachstates(plotfolder)
#fig_4_results(plotfolder)
#fig_5_transfer(plotfolder)
fig_3_auth_conftables(plotfolder)
fig_6_conftables(plotfolder)
