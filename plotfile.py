import matplotlib.pyplot as pl
import pickle
from PIL import Image
import matplotlib.pyplot as pl
import plotTools as pt
import seaborn as sns
import pandas as pd



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


def fig_8_percentage_results(plotfolder):

    all_results_df = pd.DataFrame(columns = ['train_site', 'test_site','corr-coeff', 'f1', 'nmi', 'model_type'])
    modelnames = []
    for ti, trainsite in enumerate(['nbn', 'duck', 'nbn_duck']):
        if trainsite == 'nbn' or trainsite == 'duck':
            modelnames = ['resnet512_earlystop_']
        if trainsite == 'nbn_duck':

            for addsite in ['nbn', 'duck']:
                for percentage in ['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95']:
                    modelname = 'resnet512_add{}_percentage_{}_'.format(addsite, percentage)
                    modelnames.append(modelname)
            modelnames.append('resnet512_earlystop_')

        #modelnames = ['resnet512_five_aug_']
        ########
        plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
        out_folder = 'model_output/train_on_{}/'.format(trainsite)

        skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
        if ti == 0:
            all_results_df = skc.gen_skill_df(ensemble = True)
            if trainsite == 'nbn':
                all_results_df.loc[all_results_df.model_type == 'resnet512_earlystop_', 'model_type']= 'resnet512_nbn'
        if ti >0:
            all_results_df = pd.concat((all_results_df, skc.gen_skill_df(ensemble=True)))
            if trainsite == 'duck':
                all_results_df.loc[all_results_df.model_type == 'resnet512_earlystop_', 'model_type'] = 'resnet512_duck'

    relabel_inds = {'resnet512_nbn': 0, 'resnet512_addduck_percentage_0.05_':1, 'resnet512_addduck_percentage_0.1_':2, 'resnet512_addduck_percentage_0.25_':3,
     'resnet512_addduck_percentage_0.5_':4,'resnet512_addduck_percentage_0.75_':5,'resnet512_addduck_percentage_0.9_':6,'resnet512_addduck_percentage_0.95_':7,
     'resnet512_earlystop_':8, 'resnet512_addnbn_percentage_0.95_':9, 'resnet512_addnbn_percentage_0.9_':10,'resnet512_addnbn_percentage_0.75_':11,'resnet512_addnbn_percentage_0.5_':12,
     'resnet512_addnbn_percentage_0.25_':13,'resnet512_addnbn_percentage_0.1_':14, 'resnet512_addnbn_percentage_0.05_':15, 'resnet512_duck':16}

    labels = ['100 \n0', '100 \n5', '100 \n10', '100 \n25', '100 \n50', '100 \n75', '100 \n90', '100 \n95', '100 \n100',
              '95 \n 100', '90 \n 100', '75 \n 100', '50 \n 100', '25 \n 100', '10 \n 100', '5 \n 100', '0 \n 100']

    for modelname in relabel_inds.keys():
        all_results_df.loc[all_results_df.model_type == modelname, 'num'] = relabel_inds[modelname]



    palette = {'both':'black', 'nbn':'red', 'duck':'blue'}
    model = 'resnet512_stretched_'
    fig, ax = pl.subplots(1,1, sharex = True, tight_layout = {'rect':[0, 0, 1, 0.95]}, figsize = [8, 6])
    fig.set_size_inches(10, 2.5)
    a = sns.barplot(x = 'num', y ='f1', hue = 'test_site', data = all_results_df, ax = ax, palette = palette)
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels)
    a.legend_.remove()
    a.set_xlabel('')
    pl.suptitle('F1 Score')
    a.text(-1, -0.15, 'Nbn', color = 'red')
    a.text(-1, -0.25, 'Duck', color = 'blue')
    pl.savefig(plotfolder + 'fig8_results_bar.png')

    fig, ax = pl.subplots(1,1, sharex = True, tight_layout = {'rect':[0, 0, 1, 0.97]}, figsize = [7,5])
    a = sns.lineplot(x = 'num', y ='f1', hue = 'test_site', data = all_results_df, ax = ax, palette = palette, ci = None)
    a.set_yticks([0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9,1])
    pl.grid()
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels)
    a.legend_.remove()
    a.set_xlabel('')
    a.set_ylabel('')
    ax.set_ylim(0,1)
    ax.text(-1.5, -0.05, 'Nbn', color = 'red')
    ax.text(-1.5, -0.08, 'Duck', color = 'blue')
    ax.plot((8,8), (0,1), '--k')
    ax.text(1, 0.9, "Add Duck", fontsize = 14, fontweight = 'bold')
    ax.text(10, 0.9, "Add Nbn", fontsize = 14, fontweight = 'bold')
    pl.suptitle('F1 Score')
    pl.savefig(plotfolder + 'fig8_results_line.png')



def fig_4_results(plotfolder):

    modelnames = ['resnet512_five_aug_']
    testsites = ['nbn', 'duck', 'both']
    for ti, trainsite in enumerate(['nbn', 'duck', 'nbn_duck']):
        plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
        out_folder = 'model_output/train_on_{}/'.format(trainsite)
        skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
        if ti == 0:
            all_results_df = skc.gen_skill_df(testsites = testsites, ensemble = True)
        if ti >0:
            all_results_df = pd.concat((all_results_df, skc.gen_skill_df(testsites = testsites, ensemble=True)))

    sns.set_color_codes('bright')
    fig, ax = pl.subplots(1,1, sharex = True, tight_layout = {'rect':[0, 0, 1, 0.90]})
    fig.set_size_inches(3, 3)
    metric = 'f1'
    a = sns.barplot(x = 'train_site', y =metric, hue = 'test_site', data = all_results_df, ax = ax, palette = {'both':'black', 'nbn':'salmon', 'duck':'blue'})
    #a = sns.barplot(x = metric, y ='train_site', hue = 'test_site', data = all_results_df[all_results_df.model_type == model], ax = ax[mi], palette = {'b', 'salmon'}, )
    a.legend_.remove()
    a.grid()
    ax.set_ylabel('')
    ax.set_ylim((0, 1))
    pl.suptitle('F1 Score')
    ax.set_xticklabels(['Nbn', 'Duck', 'Combined'])
    ax.set_xlabel('Train Site')
    pl.savefig(plotfolder + 'fig4_results_bar.png')


def fig_5_conftables(plotfolder):
    test_type = 'val' #'transfer' or 'val', tells the function which file to pull
    testsites = [['nbn', 'duck'], ['nbn','duck'], ['nbn', 'duck']]
    modelnames = ['resnet512_five_aug_']
    #The ensemble switch will show the average (True) or the best performing model (False)
    for ti, trainsite in enumerate(['nbn_duck','nbn', 'duck']):
        out_folder = 'model_output/train_on_{}/'.format(trainsite)
        skc = pt.skillComp(modelnames, plotfolder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
        plot_fname = 'fig_5_conftable_{}.png'.format(trainsite)
        _ = skc.gen_conf_matrix(test_type, testsites[ti], plot_fname, average = True)


plotfolder = '/home/aquilla/aellenso/Research/DeepBeach/resnet_manuscript/plots/'
# fig_2_beachstates(plotfolder)
fig_8_percentage_results(plotfolder)
# fig_4_results(plotfolder)
# fig_5_conftables(plotfolder)
