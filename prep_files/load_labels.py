import pandas as pd
import sklearn.metrics as metrics
import plotTools as pt
import matplotlib.pyplot as pl

states = ['Ref', 'LTT-B', 'TBR-CD', 'RBB-E', 'LBT-FG']
person_name = 'JS'
fig, ax = pl.subplots(2,1, sharex = True, figsize = [7,5])
for ti,testsite in enumerate(['nbn', 'duck']):
    labels_df = pd.read_pickle('labels/{}_daytimex_labels_df.pickle'.format(testsite))

    xls = pd.ExcelFile('labels/ValidationLabels_{}.xlsx'.format(person_name))
    val_js = pd.read_excel(xls, 'nbn')
    for sheet in ['duck', 'nbn_addl', 'duck_addl']:
        new_sheet = pd.read_excel(xls, sheet)
        val_js = pd.concat((val_js,new_sheet))

    col_names = val_js.columns
    val_js = val_js.drop(columns = col_names[2:])

    all_pids = labels_df.pid.values
    true = []
    person = []
    for val_pid, person_label in zip(val_js.PID, val_js.Label):
        if testsite in val_pid:
            if person_label < 5:
                datenum = val_pid.split('.')[-1]
                try:
                    pid = [aa for aa in all_pids if datenum in aa][0]
                except IndexError:
                    print('One index error')
                    continue
                truth = labels_df[labels_df.pid == pid].label.values[0]
                truth = states.index(truth)
                person.append(person_label)
                true.append(truth)

    conf_matrix = metrics.confusion_matrix(true, person)
    f1 = metrics.f1_score(true, person, average = 'weighted')

    sk = pt.skillComp([], '/home/server/pi/homes/aellenso/Research/DeepBeach/python/ResNet/prep_files/plots/', '/home/', 'duck')

    sk.confusionTable(conf_matrix, states, fig, ax[ti], testsite, testsite + ' F1: {0:.2f}'.format(f1), ensemble = False)
    ax[ti].set_ylabel('Ashley')

ax[0].set_xlabel('')
ax[1].set_xlabel(person_name)
fig.suptitle('Comp with {}'.format(person_name))
fig.savefig('prep_files/plots/compwith{}'.format(person_name))
