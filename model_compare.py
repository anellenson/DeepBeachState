import matplotlib.pyplot as pl
import plotTools as pt
import pickle
import seaborn as sns
import os
import pandas as pd

all_results_df = pd.DataFrame(columns = ['train_site', 'test_site','corr-coeff', 'f1', 'nmi', 'model_type'])
modelnames = ['resnet512_earlystop_', 'resnet512_five_aug_']
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
fig, ax = pl.subplots(3,1, sharex = True, tight_layout = {'rect':[0, 0, 1, 0.90]})
fig.set_size_inches(5, 10)
for mi,metric in enumerate(['f1', 'corr-coeff', 'nmi']):
    a = sns.barplot(x = 'train_site', y =metric, hue = 'test_site', data = all_results_df[all_results_df.model_type == 'resnet512_five_aug_'], ax = ax[mi], palette = {'both':'black', 'nbn':'salmon', 'duck':'blue'})
    #a = sns.barplot(x = metric, y ='train_site', hue = 'test_site', data = all_results_df[all_results_df.model_type == model], ax = ax[mi], palette = {'b', 'salmon'}, )
    a.legend_.remove()
    a.grid()
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
#

test_type = 'val' #'transfer' or 'val', tells the function which file to pull
testsites = [['nbn', 'duck'], ['nbn', 'duck'], ['nbn', 'duck']]
modelnames = [['resnet512_five_aug_']]
for modelname in modelnames:
    #The ensemble switch will show the average (True) or the best performing model (False)
    for ti, trainsite in enumerate(['nbn', 'duck', 'nbn_duck']):
        plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
        out_folder = 'model_output/train_on_{}/'.format(trainsite)
        skc = pt.skillComp(modelname, plot_folder, out_folder, trainsite, numruns = 10, valfile = 'cnn_preds')
        best_model = skc.gen_conf_matrix(test_type, testsites[ti], '{}conf_matrix'.format(modelname), average = False)

best_model_name = modelnames[0]
best_ensemble_member = best_model[0]
for test_site in ['duck', 'nbn']:
    for metric in ['f1', 'nmi', 'corr-coeff']:
        print('{} scores for {}'.format(metric, test_site))

        mean = results_df[(results_df.model_type == best_model_name) & (results_df.test_site == test_site)][metric].mean()
        std = results_df[(results_df.model_type == best_model_name) & (results_df.test_site == test_site)][metric].std()

        print('mean is {0:0.2f} +/- {1:0.2f}'.format(mean, std))

    print('Best Performing Model for {0} is run {1}, \n ============== \n f1: {2:0.2f} \n nmi : {3:0.2f} \n corr-coeff: {4:0.2f}'.format(test_site, best_ensemble_member, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member].f1, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member].nmi, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member]['corr-coeff']))
