import pickle
from sklearn import manifold
import matplotlib.pyplot as pl
from sklearn import decomposition
import numpy as np
import pandas as pd

with open('model_output/duck/pure_images_h512_w512_more_labels_run3_testing_weights.pickle', 'rb') as f:
    weightpick = pickle.load(f, encoding = 'latin')
class_names = ['Ref', 'LTT/B', 'TBR/CD','RBB/E', 'LBT/FG']
fnames = weightpick['pids']
weights = weightpick['allweights']
weights_array = np.zeros((len(weights), weights[0].shape[0]*weights[0].shape[1]))
for wi,ww in enumerate(weights):
    ww = ww.ravel()
    weights_array[wi, :] = ww
#pca = decomposition.PCA(n_components = 50)
#pca_weights = pca.fit_transform(weights_array)
#pca.explained_variance_ratio_


embedded_weights = manifold.TSNE(n_components=2, perplexity = 10).fit_transform(weights_array)
labels_df = pd.read_pickle('labels/pure_label_df.pickle')
labels = [class_names.index(labels_df[labels_df.pid==pid].label.values) for pid in fnames if pid in labels_df.pid.values]
missing_inds = [i for i,pid in enumerate(fnames) if pid not in labels_df.pid.values]
lined_up_weights = np.delete(embedded_weights, missing_inds, axis = 0)
pl.figure()
pl.scatter(lined_up_weights[:,0], lined_up_weights[:,1])

