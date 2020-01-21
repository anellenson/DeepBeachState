import pandas as pd
import os
import matplotlib.pyplot as pl
import numpy as np
from sklearn import metrics
import seaborn as sns
from ResultPlot import ConfResultPlot


def calc_mean(conf_dt):
    right = np.diagonal(conf_dt.values)
    class_acc = right/np.sum(conf_dt, axis =1)
    meanclassacc = np.mean(class_acc)
    all_acc = np.append(class_acc, meanclassacc)
    return all_acc


def load_df(dirname, classes):
    files = os.listdir(dirname)
    meanclasses = [cc for cc in classes]
    meanclasses.append('meanacc')
    results_df = pd.DataFrame(columns = meanclasses, index = files)
    for ff in files:
        try:
            conf_dt = pd.read_pickle(dirname + ff)
        except IOError:
            continue
        all_acc = calc_mean(conf_dt)
        results_df.loc[ff][:] = all_acc

    results_df = results_df.sort_values(by = 'meanacc',ascending = False)
    return results_df

def cat_results(class_names, modelname, modeldir):
    files = os.listdir(modeldir + modelname)
    full_array_vals = np.zeros((len(files), len(class_names), len(class_names)))
    for fi, file in enumerate(files):
        conf_dt = pd.read_pickle(modeldir + '/' + modelname + '/' + file)
        confusion_matrix = conf_dt.values
        confusion_matrix = (confusion_matrix.T / np.sum(confusion_matrix, axis=1)).T
        full_array_vals[fi,:,:] = confusion_matrix

    return full_array_vals


def return_predictions(conf_dt):
    true_labels = []
    pred_labels = []
    for ri, row in enumerate(conf_dt.values):
        true_labels = true_labels + [ri] * int(np.sum(row))
        for ci, column in enumerate(row):
            pred_labels = pred_labels + [ci] * int(column)

    return true_labels, pred_labels

def gen_skill_score(conf_dt):
    true_labels, pred_labels = return_predictions(conf_dt)
    f1 = metrics.f1_score(true_labels, pred_labels, average='weighted')
    corrcoeff = metrics.matthews_corrcoef(true_labels, pred_labels)
    nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)
    return f1, corrcoeff, nmi

def model_score(path):
    '''
    loads the confidence tables, calls on gen_skill_score to calculate score values,
    and returns a dataframe where the index is the models (ensemble members) and the columns
    are the scores
    '''
    files = os.listdir(path)
    results_df = pd.DataFrame(columns=['f1', 'corrcoeff', 'nmi'], index=files)
    for file in files:
        conf_dt = pd.read_pickle(path + file)
        results_df.loc[file] = gen_skill_score(conf_dt)

    return results_df


def unpickle(metric_filename):
    import pickle
    with open(metric_filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        model_mets = u.load()

    return model_mets

modelnames = ['train_on_duck', 'train_on_nbn']
test_site = 'duck'
plot_outdir = r'C:\Users\z3530791\DeepBeach\GitHub\python\ResNet\plots\comparison_plots\\'
class_names = ['Ref', 'LTT/B', 'TBR/CD','RBB/E', 'LBT/FG']
all_mean_acc = []
std_all_models = pd.DataFrame(index = modelnames, columns = class_names + ['meanacc'])
mean_all_models = pd.DataFrame(index = modelnames, columns = class_names + ['meanacc'])
class_acc_df = pd.DataFrame()
score_df = pd.DataFrame()
for mi,model in enumerate(modelnames):
    df_models = load_df('confusion_table_results/{}/'.format(test_site) + model + '/', class_names) #Returns the average for each class
    df_models['model'] = model
    class_acc_df = pd.concat((class_acc_df, df_models))

    score_models = model_score('confusion_table_results/{}/'.format(test_site) + model + '/')
    score_models['model'] = model
    score_df = pd.concat((score_df, score_models))


#Plot per class accuracy
fig, axes = pl.subplots(3,2,tight_layout = {'rect':[0,0,1,0.95]}, sharex = True)
fig.set_size_inches(4,6)
for ci, state in enumerate(class_names):
    ax = axes.ravel('F')[ci]
    sns.catplot("model", state, data = class_acc_df, palette = 'muted', ax = ax, kind = 'bar')
    ax.set_ylim((0, 1))
    #ax.set_xticklabels(['duck', 'nbn', 'combined'])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_xlabel('Trained On')
fig.suptitle('Tested On {}: Per Class Acc'.format(test_site))
pl.figure(fig.number)
pl.savefig(plot_outdir+'tested_{}_perclassacc.png'.format(test_site), bbox_inches='tight')
#pl.show()

fig, axes = pl.subplots(3,1,tight_layout = {'rect':[0,0,1,0.95]}, sharex = True)
fig.set_size_inches(3,6)
for ci, score in enumerate(['f1','corrcoeff', 'nmi']):
    ax = axes.ravel('F')[ci]
    sns.catplot("model", score, data = score_df, palette = 'muted', ax = ax, kind = 'bar')
    ax.set_ylim((0,1))
    #ax.set_xticklabels(['duck', 'nbn', 'combined'])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_xlabel('Trained On')
fig.suptitle('Tested On {}: Global Skill Scores'.format(test_site))
pl.figure(fig.number)
pl.savefig(plot_outdir+'tested_{}_globalskill.png'.format(test_site), bbox_inches='tight')


modeldir = 'confusion_table_results/{}/'.format(test_site)

all_conf_vals = []
for model in modelnames:
    confvals = cat_results(class_names, model, modeldir)
    all_conf_vals.append(confvals)

confplotname = 'plots/conftables_test_at_{}.png'.format(test_site)
cp = ConfResultPlot(class_names)
title = 'Tested at {}'.format(test_site)
cp.plot_conf_dt_mean_and_var(modelnames, confplotname, all_conf_vals, title)



