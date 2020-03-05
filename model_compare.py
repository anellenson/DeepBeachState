import matplotlib.pyplot as pl
import plotTools as pt
import pickle
import seaborn as sns
import os

trainsite = 'nbn'
modelnames = os.listdir('resnet_models/train_on_{}/'.format(trainsite))
modelnames = ['resnet_noise_0', 'resnet_noise_1', 'resnet_noise_2', 'resnet_noise_3', 'resnet_noise_4', 'resnet_noise_5']

####traininfo
for model in modelnames:

    with open('model_output/train_on_{}/{}/train_specs.pickle'.format(trainsite,model), 'rb') as f:
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

skc.gen_conf_matrix()
