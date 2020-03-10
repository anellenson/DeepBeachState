import matplotlib.pyplot as pl
import plotTools as pt
import pickle
import seaborn as sns
import os

trainsite = 'nbn_duck'
modelnames = os.listdir('resnet_models/train_on_{}/'.format(trainsite))
modelnames = ['resnet_noaug', 'resnet_five_aug', 'resnet_three_aug']

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
results_df = skc.gen_skill_df()

sns.set_color_codes('pastel')
fig, ax = pl.subplots(1,3, sharey = True, tight_layout = {'rect':[0, 0, 1, 0.90]})
fig.set_size_inches(10, 2.5)
for mi,metric in enumerate(['f1', 'corr-coeff', 'nmi']):
    a = sns.barplot(x = metric, y ='model_type', hue = 'test_site', data = results_df, ax = ax[mi])
    a.legend_.remove()
    a.grid()
pl.xticks(rotation = 45)
pl.suptitle('Trained on {}, tested on duck/nbn'.format(trainsite))
pl.savefig(plot_folder + '{}_SkillScore.png'.format(model))

best_model = skc.gen_conf_matrix(ensemble = False)



for test_site in ['duck', 'nbn']:
    for metric in ['f1', 'corr-coeff', 'nmi']:
        print('{} scores for {}'.format(metric, test_site))

        mean = results_df[(results_df.model_type == 'resnet_five_aug') & (results_df.test_site == test_site)][metric].mean()
        std = results_df[(results_df.model_type == 'resnet_five_aug') & (results_df.test_site == test_site)][metric].std()

        print('mean is {} +/- {}'.format(mean, std))

    print('Best Performing Model for {0}, \n ============== \n f1: {1:0.2f} \n nmi : {2:0.2f} \n corr-coeff: {3:0.2f}'.format(test_site, results_df[(results_df.test_site == test_site) & (results_df.model_type == 'resnet_five_aug')].iloc[best_model[1]].f1, results_df[(results_df.test_site == test_site) & (results_df.model_type == 'resnet_five_aug')].iloc[best_model[1]].nmi, results_df[(results_df.test_site == test_site) & (results_df.model_type == 'resnet_five_aug')].iloc[best_model[1]]['corr-coeff']))
