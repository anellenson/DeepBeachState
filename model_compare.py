import matplotlib.pyplot as pl
import plotTools as pt
import pickle
import seaborn as sns
import os

trainsite = 'nbn_duck'
modelnames = os.listdir('resnet_models/train_on_{}/'.format(trainsite))
modelnames = ['resnet512_noaug', 'resnet512_five_aug']

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


best_model_name = modelnames[1]
best_ensemble_member = best_model[1]
for test_site in ['duck', 'nbn']:
    for metric in ['f1', 'nmi', 'corr-coeff']:
        print('{} scores for {}'.format(metric, test_site))

        mean = results_df[(results_df.model_type == best_model_name) & (results_df.test_site == test_site)][metric].mean()
        std = results_df[(results_df.model_type == best_model_name) & (results_df.test_site == test_site)][metric].std()

        print('mean is {0:0.2f} +/- {1:0.2f}'.format(mean, std))

    print('Best Performing Model for {0} is run {1}, \n ============== \n f1: {2:0.2f} \n nmi : {3:0.2f} \n corr-coeff: {4:0.2f}'.format(test_site, best_ensemble_member, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member].f1, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member].nmi, results_df[(results_df.test_site == test_site) & (results_df.model_type == best_model_name)].iloc[best_ensemble_member]['corr-coeff']))
