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



