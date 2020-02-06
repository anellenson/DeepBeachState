import matplotlib.pyplot as pl
import plotTools as pt
import pickle
import seaborn as sns
import os


modelnames = ['inception_resnet_aug_fulltrained', 'inception_resnet_no_aug_fulltrained', 'resnet_aug_fulltrained',  'resnet_no_aug_fulltrained', 'resnet50_aug_fulltrained', 'resnet50_no_aug_fulltrained', 'mobilenet_aug_fulltrained', 'mobilenet_no_aug_fulltrained']

####traininfo
for model in modelnames:

    with open('model_output/train_on_nbn/{}/train_specs.pickle'.format(model), 'rb') as f:
        trainInfo = pickle.load(f)

    plot_fname = 'plots/train_on_nbn/train_info/{}.png'.format(model)


    pt.trainInfo(trainInfo['val_acc'], trainInfo['train_acc'], trainInfo['val_loss'], trainInfo['train_loss'], plot_fname, model)


########
plot_folder = 'plots/train_on_nbn/skill_compare/'
out_folder = 'model_output/train_on_nbn/'

skc = pt.skillComp(modelnames, plot_folder, out_folder)
results_df = skc.gen_skill_df()

sns.set_color_codes('pastel')
fig, ax = pl.subplots(1,3, sharey = True, tight_layout = {'rect':[0, 0, 1, 0.90]})
for mi,metric in enumerate(['f1', 'corr-coeff', 'nmi']):
    a = sns.barplot(x = metric, y ='model_type', hue = 'test_site', data = results_df, ax = ax[mi])
    a.legend_.remove()
    a.grid()
pl.xticks(rotation = 45)

skc.gen_conf_matrix()
