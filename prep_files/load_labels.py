import pandas as pd
import pickle
import sklearn.metrics as metrics
import plotTools as pt
import matplotlib.pyplot as pl
import numpy as np

def load_xls(testsite):

    states = ['Ref', 'LTT-B', 'TBR-CD', 'RBB-E', 'LBT-FG']
    with open('labels/{}_daytimex_valfiles.final.pickle'.format(testsite), 'rb') as f:
        valfiles = pickle.load(f)

    labels_df = pd.read_pickle('labels/{}_daytimex_labels_df.pickle'.format(testsite))
    vallabels_df = pd.DataFrame(index = valfiles, columns = ['AE', 'KS', 'KS1', 'GW', 'JS'])

    for person_name in ['KS', 'KS1', 'GW', 'JS']:

        xls = pd.ExcelFile('labels/ValidationLabels_{}.xlsx'.format(person_name))
        val_js = pd.read_excel(xls, testsite)
        try:
            new_sheet = pd.read_excel(xls, '{}_addl'.format(testsite))
            val_js = pd.concat((val_js,new_sheet))
        except:
            pass

        col_names = val_js.columns
        val_js = val_js.drop(columns = col_names[2:])

        for val_pid, person_label in zip(val_js.PID, val_js.Label):
            if person_label < 5:
                datenum = val_pid.split('.')[-1]
                try:
                    pid = [aa for aa in valfiles if datenum in aa][0]
                except IndexError:
                    print(datenum)
                    continue
                vallabels_df[person_name].loc[pid] = person_label

    for pid in vallabels_df.index:
        ae_label = labels_df[labels_df.pid == pid].label.values[0]
        ae_label = states.index(ae_label)
        vallabels_df['AE'].loc[pid] = ae_label

    vallabels_df.to_pickle('labels/{}_daytimex_valfiles_allauthors_df.pickle'.format(testsite))


states = ['Ref', 'LTT-B', 'TBR-CD', 'RBB-E', 'LBT-FG']
fig, ax = pl.subplots(2,1, tight_layout = {'rect':[0,0,1, 0.95]}, sharex = True)
for ti, testsite in enumerate(['nbn', 'duck']):
    f1_list = []
    for pi,person_name in enumerate(["GW", "KS", "KS1", 'JS']):

        vallabels_df = pd.read_pickle('labels/{}_daytimex_valfiles_allauthors_df.pickle'.format(testsite))
        labels_1 = vallabels_df['AE']
        labels_2 = vallabels_df[person_name]

        labels_2 = labels_2[labels_1.notna()]
        labels_1 = labels_1[labels_1.notna()]

        vallabels1 = [vv for vv in labels_1[labels_2.notna()].values]
        vallabels2 = [vv for vv in labels_2[labels_2.notna()].values]

        if not vallabels2  == []:
            if pi == 0:
                conf_matrix = metrics.confusion_matrix(vallabels1, vallabels2)
                conf_matrix = np.expand_dims(conf_matrix, axis = 0)
            if pi>0:
                conf_matrix_1 = metrics.confusion_matrix(vallabels1, vallabels2)
                conf_matrix_1 = np.expand_dims(conf_matrix_1, axis = 0)
                conf_matrix = np.concatenate((conf_matrix, conf_matrix_1), axis = 0)

            f1 = metrics.f1_score(vallabels1, vallabels2, average = 'weighted')
            f1_list.append(f1)


    sk = pt.skillComp([], '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/prep_files/plots/', '/home/', testsite)

    sk.confusionTable(conf_matrix, states, fig, ax[ti], testsite, testsite + ' F1: {0:0.2f}'.format(f1, np.std(f1_list)), ensemble = True)
    ax[0].set_ylabel('AE')
    ax[1].set_ylabel('AE')
    ax[1].set_xlabel('All')

    fig.suptitle('Comp with all'.format(person_name))
    fig.savefig('/home/aquilla/aellenso/Research/DeepBeach/resnet_manuscript/plots/fig7_conftable_coauthors.png')
