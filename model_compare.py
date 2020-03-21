import matplotlib.pyplot as pl
import plotTools as pt
import pickle
import seaborn as sns
import os
import pandas as pd

all_results_df = pd.DataFrame(columns = ['train_site', 'test_site','corr-coeff', 'f1', 'nmi', 'model_type'])

for ti, trainsite in enumerate(['duck', 'nbn', 'nbn_duck']):
    modelnames = os.listdir('resnet_models/train_on_{}/'.format(trainsite))
    modelnames = ['resnet512_five_aug', 'resnet512_noaug']

    ####traininfo
    for model in modelnames:

        with open('model_output/train_on_{}/{}_0/train_specs.pickle'.format(trainsite,model), 'rb') as f:
            trainInfo = pickle.load(f)

        plot_fname = 'plots/train_on_{}/train_info/{}.png'.format(trainsite,model)


        pt.trainInfo(trainInfo['val_acc'], trainInfo['train_acc'], trainInfo['val_loss'], trainInfo['train_loss'], plot_fname, model)


    ########
    plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
    out_folder = 'model_output/train_on_{}/'.format(trainsite)

    skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite)
    if ti == 0:
        all_results_df = skc.gen_skill_df()
    if ti >0:
        all_results_df = pd.concat((all_results_df, skc.gen_skill_df()))


sns.set_color_codes('bright')
model = 'resnet512_five_aug'
fig, ax = pl.subplots(1,3, sharey = True, tight_layout = {'rect':[0, 0, 1, 0.90]})
fig.set_size_inches(10, 2.5)
for mi,metric in enumerate(['f1', 'corr-coeff', 'nmi']):
    a = sns.barplot(x = metric, y ='train_site', hue = 'test_site', data = all_results_df[all_results_df.model_type == model], ax = ax[mi], palette = {'b', 'salmon'}, )
    a.legend_.remove()
    a.grid()
pl.xticks(rotation = 45)
ax[0].set_xlim((0, 0.8))
ax[1].set_xlim((0, 0.8))
ax[2].set_xlim((0, 0.6))
ax[2].set_ylabel('')
ax[1].set_ylabel('')
ax[0].set_yticklabels(['Duck', 'Nbn', 'Combined'])
ax[0].set_xlabel('F1')
ax[1].set_xlabel('Corr-Coeff')
ax[2].set_xlabel('NMI')
ax[0].set_ylabel('Train Site')
pl.suptitle('CNN Skill', fontsize = 14, fontname = 'Helvetica')
pl.savefig('/home/server/pi/homes/aellenso/Research/DeepBeach/resnet_manuscript/plots/overall_skillscore.png')


for trainsite in ['duck', 'nbn', 'nbn_duck']:
    plot_folder = 'plots/train_on_{}/skill_compare/'.format(trainsite)
    out_folder = 'model_output/train_on_{}/'.format(trainsite)

    skc = pt.skillComp(modelnames, plot_folder, out_folder, trainsite)

    best_model = skc.gen_conf_matrix(ensemble = False)


best_model_name = modelnames[0]
best_ensemble_member = best_model[0]
for test_site in ['duck', 'nbn']:
    for metric in ['f1', 'nmi', 'corr-coeff']:
        print('{} scores for {}'.format(metric, test_site))

        mean = results_df[(results_df.model_type == best_model_name) & (results_df.test_site == test_site)][metric].mean()
        std = results_df[(results_df.model_type == best_model_name) & (results_df.test_site == test_site)][metric].std()

        print('mean is {0:0.2f} +/- {1:0.2f}'.format(mean, std))

    print('Best Performing Model for {0} is run {1}, \n ============== \n f1: {2:0.2f} \n nmi : {3:0.2f} \n corr-coeff: {4:0.2f}'.format(test_site, best_ensemble_member, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member].f1, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member].nmi, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member]['corr-coeff']))
