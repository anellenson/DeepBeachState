from __future__ import division
from matplotlib import pyplot as pl
import matplotlib
import numpy as np
from sklearn import metrics
import pandas as pd
import pickle
import matplotlib.ticker as ticker
import math


class skillComp():

    def __init__(self, modelnames, plot_folder, out_folder, trainsite, numruns =10, valfile = 'cnn_preds'):
        self.modelnames = modelnames
        self.plot_folder = plot_folder
        self.out_folder = out_folder
        self.trainsite = trainsite
        self.numruns = numruns
        self.valfile = valfile



    def gen_skill_score(self, true_labels, pred_labels):

        f1 = metrics.f1_score(true_labels, pred_labels, average='weighted')
        corrcoeff = metrics.matthews_corrcoef(true_labels, pred_labels)
        nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)

        return f1, corrcoeff, nmi

    def gen_acc(self, true_labels, pred_labels):


        f1 = metrics.f1_score(true_labels, pred_labels, average='weighted')
        corrcoeff = metrics.matthews_corrcoef(true_labels, pred_labels)
        nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)

        return f1, corrcoeff, nmi



    def gen_skill_df(self, testsites = ['duck', 'nbn','both'], ensemble=True):
        '''
        This will produce a results dataframe with F1, NMI, and Correlation Coefficient
        The dataframe will have the 'test site' and 'model' for plotting in seaborn
        It uses 'gen skill score'

        '''
        results_df = pd.DataFrame(columns = ['train_site', 'test_site','corr-coeff', 'f1', 'nmi', 'model_type'])

        for model in self.modelnames:

            if ensemble:
                for run in range(self.numruns):
                    with open(self.out_folder +'{}{}/{}.pickle'.format(model, run, self.valfile), 'rb') as f:
                        predictions = pickle.load(f)

                    for testsite in testsites:
                        if testsite == 'both':
                            cnn_preds =  []
                            true = []
                            for site in ['duck', 'nbn']:
                                cnn_preds +=  predictions['{}_CNN'.format(site)]
                                true +=  predictions['{}_truth'.format(site)]
                        else:
                            cnn_preds = predictions['{}_CNN'.format(testsite)]
                            true =  predictions['{}_truth'.format(testsite)]

                        #Will throw an error when this is no longer a tensor
                        if testsite not in self.valfile:
                            cnn_preds = [cc.item() for cc in cnn_preds]
                            true = [tt.item() for tt in true]

                        f1,corrcoeff,nmi = self.gen_skill_score(true, cnn_preds)

                        results = {'train_site':self.trainsite, 'model_type':model, 'f1':f1, 'nmi':nmi, 'corr-coeff':corrcoeff,'test_site':testsite}
                        results_df = results_df.append(results, ignore_index = True)

            else:
                with open(self.out_folder +'{}/{}.pickle'.format(model, self.valfile), 'rb') as f:
                        predictions = pickle.load(f)

                for testsite in ['duck','nbn', 'both']:
                    if testsite == 'both':
                        cnn_preds =  []
                        true = []
                        for testsite in ['duck', 'nbn']:
                            cnn_preds +=  predictions['{}_CNN'.format(testsite)]
                            true +=  predictions['{}_truth'.format(testsite)]

                    cnn_preds = predictions['{}_CNN'.format(testsite)]
                    true =  predictions['{}_truth'.format(testsite)]


                    cnn_preds = [cc.item() for cc in cnn_preds]
                    true = [tt.item() for tt in true]

                    f1,corrcoeff,nmi = self.gen_skill_score(true, cnn_preds)

                    results = {'train_site':self.trainsite, 'model_type':model, 'f1':f1, 'nmi':nmi, 'corr-coeff':corrcoeff,'test_site':testsite}
                    results_df = results_df.append(results, ignore_index = True)


        self.results_df = results_df

        return results_df


    def confusionTable(self, confusion_matrix, class_names, fig, ax, testsite, title, ensemble = True):

        cmap_dict = {'duck':'Blues', 'nbn':'Reds'}
        cmap = cmap_dict[testsite]

        if ensemble:
            sum_confusion_matrix = confusion_matrix.sum(axis = 0)
            v = sum_confusion_matrix.sum(axis = 1)
            confusion_matrix = sum_confusion_matrix/v[:,]
            print("Trained at {}, tested at {}".format(self.trainsite, testsite))
            print(sum_confusion_matrix)
            print('Total images per state {}'.format(sum_confusion_matrix.sum(axis=1)[0]))


            for ci, state in enumerate(['Ref', 'LTT', 'TBR', 'RBB', 'LBT']):

                print("num images confused as {}: {}".format(state, sum_confusion_matrix.sum(axis = 0)[ci]/sum_confusion_matrix.sum()))

            print("Overall accuracy is {}".format(sum_confusion_matrix.diagonal().sum()/sum_confusion_matrix.sum()))
            class_acc = confusion_matrix.diagonal()
            im = ax.pcolor(confusion_matrix, cmap = cmap, vmin = 0, vmax =1)


        else:
            confusion_matrix = confusion_matrix/np.sum(confusion_matrix, axis = 1)
            class_acc = confusion_matrix.diagonal()
            im = ax.pcolor(confusion_matrix, cmap = cmap, vmin = 0, vmax = 1)


        for row in np.arange(confusion_matrix.shape[0]):
            for col in np.arange(confusion_matrix.shape[1]):
                if row == col:
                    if ensemble:
                        mean = class_acc[row]
                        if mean >= 0.5:
                            color = 'white'
                        else:
                            color = 'black'
                        ax.text(col +0.05, row+0.65, '{0:d}%'.format(int(mean*100)), fontsize = 10, fontweight = 'bold', color = color)

                    else:
                        mean = class_acc[row]
                        print(mean)
                        if mean >= 0.5:
                            color = 'white'
                        else:
                            color = 'black'
                        ax.text(col +0.05, row+0.65, '{0:d}%'.format(int(mean*100)), fontsize = 10, fontweight = 'bold', color = color)



        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.yaxis.tick_left()
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_locator(ticker.FixedLocator([0.5, 1.5, 2.5, 3.5, 4.5]))
        ax.xaxis.set_minor_formatter(ticker.FixedFormatter(class_names))


        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_locator(ticker.FixedLocator([0.5, 1.5, 2.5, 3.5, 4.5]))
        ax.yaxis.set_minor_formatter(ticker.FixedFormatter(class_names))


        ax.set_ylabel('Truth', fontsize = 12, weight = 'bold')
        ax.set_xlabel('CNN', fontsize = 12, weight = 'bold')
        #ax.set_title('Test on {0}'.format(testsite, class_acc.mean()))
        ax.set_title(title)
        cb = fig.colorbar(im, ax = ax)
        cb.ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
        cb.set_label('Accuracy (%)')



    def load_conf_table(self, model, run, testsite, filename):

        with open(self.out_folder + '{}{}/{}.pickle'.format(model, run, filename), 'rb') as f:
            predictions = pickle.load(f)

        cnn_preds = predictions['{}_CNN'.format(testsite)]
        true =  predictions['{}_truth'.format(testsite)]

        return cnn_preds, true

    def gen_conf_matrix(self, testtype, testsites, plot_fname, average = True):
        '''

        This will generate confusion matrices for both test sites.


        '''
        class_names = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
        best_models = []


        for model in self.modelnames:
            print(model)

            fig, axes = pl.subplots(len(testsites),1, tight_layout = {'rect':[0, 0, 1, 0.95]}, sharex = True, figsize = [4,5])

            f1 = np.zeros((2,10))

            for ti,testsite in enumerate(testsites):

                if len(testsites) == 1:
                    ax = axes
                else:
                    ax = axes[ti]


                for run in range(self.numruns):


                    cnn_preds, true = self.load_conf_table(model, run, testsite, self.valfile)

                    f1[ti, run] = metrics.f1_score(true, cnn_preds, average='weighted')

                    if testtype == 'val':
                        cnn_preds = [cc.item() for cc in cnn_preds]
                        true = [cc.item() for cc in true]


                    if run == 0:
                        conf_matrix = metrics.confusion_matrix(true, cnn_preds)
                        conf_matrix = np.expand_dims(conf_matrix, axis = 0)

                    if run > 0:
                        conf_matrix_ = metrics.confusion_matrix(true, cnn_preds)
                        conf_matrix_ = np.expand_dims(conf_matrix_, axis = 0)

                        conf_matrix = np.concatenate((conf_matrix, conf_matrix_), axis = 0)

                title = 'Test at {}'.format(testsite)

                if average:
                    self.confusionTable(conf_matrix, class_names, fig, ax, testsite, title, ensemble = average)


            if not average:
                f1_mean = np.mean(f1, axis = 0) #choose the best performing model and reload the predictions to plot them.
                best_model_index = np.argwhere(f1_mean == np.max(f1_mean))[0][0]
                best_models.append(best_model_index) #return the best performing ensemble member for each model type


            if not average:

              for ti,testsite in enumerate(testsites):

                if len(testsites) == 1:
                    ax = axes
                else:
                    ax = axes[ti]          #load the data

                cnn_preds, true = self.load_conf_table(model, best_model_index, testsite,  self.valfile)
                if testtype == 'val':
                    cnn_preds = [cc.item() for cc in cnn_preds]
                    true = [cc.item() for cc in true]

                conf_matrix = metrics.confusion_matrix(true, cnn_preds)
                self.confusionTable(conf_matrix, class_names, fig, ax, testsite, title, ensemble = average)


            pl.suptitle('Confusion Matrix \n trained at {}'.format(self.trainsite))

            pl.savefig(self.plot_folder + plot_fname)
            print('Printed Confusion Matrix for {}'.format(model))

        return best_models, fig, axes


def trainInfo(val_acc, train_acc, val_loss, train_loss, plot_fname, title):

    fig = pl.figure()
    pl.clf()
    pl.subplot(121)
    pl.plot(np.arange(0,len(val_acc)), val_acc, color= 'purple', label = 'val')
    pl.plot(np.arange(0,len(val_acc)), train_acc, color = 'orange', label = 'train')
    pl.legend()
    pl.xlabel('Epoch')
    pl.title('Accuracy')

    pl.subplot(122)
    pl.plot(np.arange(0,len(val_loss)), val_loss, color = 'purple', label = 'val')
    pl.plot(np.arange(0,len(train_loss)), train_loss, color = 'orange', label = 'train')
    pl.xlabel('Epoch')
    pl.title('Loss')

    pl.suptitle(title)

    pl.savefig(plot_fname, dpi = 600)


def resize_images(cam_dir):

    from skimage.transform import rescale, resize, downscale_local_mean
    for state in ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']:
        for ii in [1, 5, 10, 15, 20]:
            with open(cam_dir + 'BackCam_{}_{}.pickle'.format(state, ii), 'rb') as f:
                cams = pickle.load(f)

            down_cams_resize_128 = []
            down_cams_downscale_128 = []
            for cam in cams:
                cam_resize = resize(cam, (128,128), preserve_range = True)
                down_cams_resize_128.append(cam_resize)

                cam_downscale = downscale_local_mean(cam, (4,4))
                down_cams_downscale_128.append(cam_downscale)

            print(len(down_cams_downscale_128))
            with open(cam_dir + 'BackCam_{}_{}_resize_128.pickle'.format(state, ii), 'wb') as f:
                pickle.dump(down_cams_resize_128, f)

            with open(cam_dir + 'BackCam_{}_{}_downscale_128.pickle'.format(state, ii), 'wb') as f:
                pickle.dump(down_cams_downscale_128, f)
