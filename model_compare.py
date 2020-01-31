import matplotlib.pyplot as pl
import plotTools as pt
import pickle
import seaborn as sns


modelnames = ['aug_pretrained_resnet50', 'aug_fulltrained_resnet50', 'no_aug_pretrained_resnet50', 'no_aug_fulltrained_resnet50']

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

fig, ax = pl.subplots(3,1)
for mi,metric in enumerate(['f1', 'corr-coeff', 'nmi']):
    sns.barplot(x = 'model', y = metric, hue = 'test_site', data = results_df, ax = ax[mi])

skc.gen_conf_matrix()
