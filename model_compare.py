import matplotlib.pyplot as pl
import plotTools as pt
import pickle
import seaborn as sns
import os
import pandas as pd

all_results_df = pd.DataFrame(columns = ['train_site', 'test_site','corr-coeff', 'f1', 'nmi', 'model_type'])
modelnames = []
for ti, trainsite in enumerate(['nbn', 'duck', 'nbn_duck']):
    if trainsite == 'nbn' or trainsite == 'duck':
        modelnames = ['resnet512_five_aug_']
    if trainsite == 'nbn_duck':

        for addsite in ['nbn', 'duck']:
            for percentage in ['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95']:
                modelname = 'resnet512_add{}_percentage_{}_'.format(addsite, percentage)
                modelnames.append(modelname)
        modelnames.append('resnet512_five_aug_')

    #modelnames = ['resnet512_five_aug_']
    ########
    plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
    out_folder = 'model_output/train_on_{}/'.format(trainsite)

    skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
    if ti == 0:
        all_results_df = skc.gen_skill_df(ensemble = True)
        if trainsite == 'nbn':
            all_results_df.loc[all_results_df.model_type == 'resnet512_five_aug_', 'model_type']= 'resnet512_nbn'
    if ti >0:
        all_results_df = pd.concat((all_results_df, skc.gen_skill_df(ensemble=True)))
        if trainsite == 'duck':
            all_results_df.loc[all_results_df.model_type == 'resnet512_five_aug_', 'model_type'] = 'resnet512_duck'

relabel_inds = {'resnet512_nbn':0, 'resnet512_addduck_percentage_0.05_':1, 'resnet512_addduck_percentage_0.1_':2, 'resnet512_addduck_percentage_0.25_':3,
 'resnet512_addduck_percentage_0.5_':4,'resnet512_addduck_percentage_0.75_':5,'resnet512_addduck_percentage_0.9_':6,'resnet512_addduck_percentage_0.95_':7,
 'resnet512_five_aug_':8, 'resnet512_addnbn_percentage_0.95_':9, 'resnet512_addnbn_percentage_0.9_':10,'resnet512_addnbn_percentage_0.75_':11,'resnet512_addnbn_percentage_0.5_':12,
 'resnet512_addnbn_percentage_0.25_':13,'resnet512_addnbn_percentage_0.1_':14, 'resnet512_addnbn_percentage_0.05_':15, 'resnet512_duck':16}

for modelname in relabel_inds.keys():
    all_results_df.loc[all_results_df.model_type == modelname, 'num'] = relabel_inds[modelname]


sns.set_color_codes('bright')
model = 'resnet512_stretched_'
fig, ax = pl.subplots(3,1, sharex = True, tight_layout = {'rect':[0, 0, 1, 0.90]})
fig.set_size_inches(10, 2.5)
for mi,metric in enumerate(['f1', 'corr-coeff', 'nmi']):
    a = sns.barplot(x = 'num', y =metric, hue = 'test_site', data = all_results_df, ax = ax[mi], palette = {'b', 'salmon'})
    #a = sns.barplot(x = metric, y ='train_site', hue = 'test_site', data = all_results_df[all_results_df.model_type == model], ax = ax[mi], palette = {'b', 'salmon'}, )
    #a.legend_.remove()
    a.grid()
pl.xticks(rotation = 45)
ax[0].set_ylim((0, 1))
ax[1].set_ylim((0, 1))
ax[2].set_ylim((0, 1))
ax[2].set_xlabel('')
ax[1].set_xlabel('')
#ax[0].set_yticklabels(['Nbn', 'Duck', 'Combined'])
ax[0].set_xlabel('F1')
ax[1].set_xlabel('Corr-Coeff')
ax[2].set_xlabel('NMI')
ax[0].set_ylabel('Train Site')
pl.suptitle('CNN Skill', fontsize = 14, fontname = 'Helvetica')
pl.savefig('/home/aquilla/aellenso/Research/DeepBeach/research_notes/Apr2020/stretched_percentages_results/percentage_overall_skillscore_addnbn.png')


test_type = 'val' #'transfer' or 'val', tells the function which file to pull
testsites = [['nbn', 'duck'], ['nbn','duck'], ['nbn', 'duck']]
#The ensemble switch will show the average (True) or the best performing model (False)
for ti, trainsite in enumerate(['nbn_duck','nbn']):
    if trainsite == 'nbn':
        modelnames = ['resnet512_five_aug_']
    if trainsite == 'nbn_duck':
        modelnames = ['resnet512_addnbn_percentage_0.05_','resnet512_addnbn_percentage_0.1_','resnet512_addnbn_percentage_0.25_','resnet512_addnbn_percentage_0.5_','resnet512_addnbn_percentage_0.75_']

    plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
    out_folder = 'model_output/train_on_{}/'.format(trainsite)

    skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite, numruns = 5, valfile = 'cnn_preds')

    best_model = skc.gen_conf_matrix(test_type, testsites[ti], average = True)

best_model_name = modelnames[0]
best_ensemble_member = best_model[0]
for test_site in ['duck', 'nbn']:
    for metric in ['f1', 'nmi', 'corr-coeff']:
        print('{} scores for {}'.format(metric, test_site))

        mean = results_df[(results_df.model_type == best_model_name) & (results_df.test_site == test_site)][metric].mean()
        std = results_df[(results_df.model_type == best_model_name) & (results_df.test_site == test_site)][metric].std()

        print('mean is {0:0.2f} +/- {1:0.2f}'.format(mean, std))

    print('Best Performing Model for {0} is run {1}, \n ============== \n f1: {2:0.2f} \n nmi : {3:0.2f} \n corr-coeff: {4:0.2f}'.format(test_site, best_ensemble_member, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member].f1, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member].nmi, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member]['corr-coeff']))
