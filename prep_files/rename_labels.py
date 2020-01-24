import datetime as dt
import pandas as pd
import os
from __builtin__ import any as b_any


labels_df = pd.read_pickle('labels/nbn_labels_cleaned_165.pickle')

new_pids= os.listdir('/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_spz')
new_pids_dict ={}

#This script will rename the files, keep the same label from the date:

for pid in labels_df.pid:
    strnum = pid.split('.')[0]
    datenum = int(strnum.split('_')[1])
    datenum = int(datenum/1E5)

    if b_any(str(datenum) in p for p in new_pids):
        new_pid = [p for p in new_pids if str(datenum) in p][0]
        new_entry = {new_pid:labels_df[labels_df.pid == pid].label.values[0]}
        new_pids_dict.update(new_entry)


#Now check for the remainder items in the new pids that aren't labelled
missing_pids = [p for p in new_pids if p not in list(new_pids_dict.keys())]

with open('missing_pids_labels.txt', 'wb') as f:
    for pid in missing_pids:
        f.write(pid + '\n')

missing_labels = pd.read_csv('prep_files/missing_pids_labels.txt')
new_pids = pd.DataFrame({'pid':list(new_pids_dict.keys()), 'label':list(new_pids_dict.values())})

rename = {'LBT':'LBT-FG','RBB':'RBB-E','Ref':'Ref','TBR':'TBR-CD','LTT':'LTT-B'}
for li,label in enumerate(missing_labels.label):
    missing_labels.iloc[li].label = rename[label]

daytimex_labels = pd.concat((new_pids,missing_labels))

#How many entries of each class?
for state in list(rename.values()):
    print(state + ': {}'.format(len(labels_df[labels_df.label == state])))
daytimex_labels.to_pickle('labels/nbn_daytimex_labels.pickle')
