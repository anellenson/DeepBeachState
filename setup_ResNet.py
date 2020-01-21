from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import preResnet as pre
import pickle
import os
import numpy as np
'''
class MyDataset(Dataset):
    def __init__(self, list_IDs, basedir, transform = None):
        self.list_IDs = list_IDs
        self.basedir = basedir
        self.transform = transform

    def __getitem__(self, index):
        ID = self.list_IDs[index].encode('ascii')

        path = self.basedir + ID

        with open(path, 'rb') as f:
            X = Image.open(f)
            X = self.transform(X)

        return X

    def __len__(self):
        return len(self.list_IDs)


from PIL import Image

gray_dir = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/north/match_nbn/'
labels_df = pd.read_pickle('labels/pure_label_df.pickle')
filenames = labels_df.pid.values
trans = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

dataset = MyDataset(filenames, gray_dir, transform=trans)
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)



mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
'''

############set up validation dataset
class_names = ['Ref','LTT-B','TBR-CD','RBB-E','LBT-FG'] #TO DO change states list to dashes from matfile
labels_df  = pd.read_pickle('labels/nbn_labels_cleaned_165.pickle')
img_folder = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/orig_gray'
files = os.listdir(img_folder)
missing_files = [ff for ff in labels_df.pid if ff not in files]
partition, labels = pre.createTrainValSets(labels_df, class_names)
valfiles = {'valfiles':partition['val']}
with open('labels/nbn_valfiles_15perclass.pickle', 'wb') as f:
    pickle.dump(valfiles, f)
print('saved val files')

trainfiles = []
for state in class_names:
    inds_class = np.where(labels_df.label == state)[0]
    train_files_for_class = [labels_df.iloc[ii].pid for ii in inds_class if labels_df.iloc[ii].pid not in valfiles]
    train_files_for_class = train_files_for_class[:75]
    trainfiles = trainfiles + train_files_for_class

with open('labels/trainfiles_nbn.pickle', 'wb') as f:
    pickle.dump(trainfiles, f)

print('saved train files')

