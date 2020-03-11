from __future__ import division
from matplotlib import pyplot as pl
import numpy as np
from sklearn import metrics
import pandas as pd
import pickle


class skillComp():

    def __init__(self, modelnames, plot_folder, out_folder, trainsite):
        self.modelnames = modelnames
        self.plot_folder = plot_folder
        self.out_folder = out_folder
        self.trainsite = trainsite


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



    def gen_skill_df(self):
        '''
        This will produce a results dataframe with F1, NMI, and Correlation Coefficient
        The dataframe will have the 'test site' and 'model' for plotting in seaborn
        It uses 'gen skill score'

        '''
        results_df = pd.DataFrame(columns = ['test_site','corr-coeff', 'f1', 'nmi', 'model_type'])

        for model in self.modelnames:
            for run in range(5):
                with open(self.out_folder +'{}_{}/cnn_preds.pickle'.format(model, run), 'rb') as f:
                    predictions = pickle.load(f)

                for testsite in ['duck', 'nbn']:
                    cnn_preds = predictions['{}_CNN'.format(testsite)]
                    true =  predictions['{}_truth'.format(testsite)]

                    #Will throw an error when this is no longer a tensor

                    cnn_preds = [cc.item() for cc in cnn_preds]
                    true = [tt.item() for tt in true]

                    f1,corrcoeff,nmi = self.gen_skill_score(true, cnn_preds)

                    results = {'model_type':model, 'f1':f1, 'nmi':nmi, 'corr-coeff':corrcoeff,'test_site':testsite}
                    results_df = results_df.append(results, ignore_index = True)


        self.results_df = results_df

        return results_df


    def confusionTable(self, confusion_matrix, class_names, fig, ax, testsite, ensemble = True):

        cmap_dict = {'duck':'Blues', 'nbn':'Reds'}
        cmap = cmap_dict[testsite]

        if ensemble:
            class_acc = confusion_matrix[0,:,:].diagonal()/np.sum(confusion_matrix[0,:,:], axis = 1)
            for matrix in confusion_matrix:
                class_acc = np.vstack((class_acc, matrix.diagonal()/np.sum(matrix, axis = 1)))

            acc = np.sum(confusion_matrix, axis = 0)/np.sum(np.sum(confusion_matrix, axis = 0), axis = 1)
            im = ax.pcolor(np.sum(confusion_matrix, axis = 0), cmap = cmap)


        else:
            confusion_matrix = confusion_matrix/np.sum(confusion_matrix, axis = 1)
            class_acc = confusion_matrix.diagonal()
            im = ax.pcolor(confusion_matrix, cmap = cmap)


        for row in np.arange(confusion_matrix.shape[0]):
            for col in np.arange(confusion_matrix.shape[1]):
                if row == col:
                    if ensemble:
                        mean = class_acc[:,row].mean()
                        std  = class_acc[:,row].std()
                        if mean >= 0.5:
                            color = 'white'
                        else:
                            color = 'black'
                        ax.text(col +0.05, row+0.65, '{0:.2f} +/- {1:.2f}'.format(mean, std), fontsize = 9, fontweight = 'bold', color = color)

                    else:
                        mean = class_acc[row]
                        if mean >= 0.5:
                            color = 'white'
                        else:
                            color = 'black'
                        ax.text(col +0.35, row+0.65, '{0:.2f}'.format(mean), fontsize = 15, fontweight = 'bold', color = color)
                # if confusion_matrix[row, col] >= 30:
                #     ax.text(col +0.35, row+0.65, str(int(confusion_matrix[row, col])), fontsize = 20, fontweight = 'bold', color = 'white')
                # if confusion_matrix[row,col] < 30:
                #     ax.text(col+0.35, row+0.65, str(int(confusion_matrix[row, col])), fontsize = 20, fontweight = 'bold')

        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.yaxis.tick_left()
        ax.set_xticklabels(class_names, fontsize = 10, weight = 'bold')
        ax.set_yticklabels(class_names, fontsize = 10, weight = 'bold')
        ax.set_ylabel('Truth', fontsize = 12, weight = 'bold')
        ax.set_xlabel('CNN', fontsize = 12, weight = 'bold')
        ax.set_title('Test on {0}'.format(testsite, class_acc.mean()))
        cb = fig.colorbar(im, ax = ax, ticks = [0, 0.2, 0.4, 0.6, 0.8, 1])
        cb.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])



    def load_conf_table(self, model, run, testsite):

        with open(self.out_folder + '{}_{}/cnn_preds.pickle'.format(model, run), 'rb') as f:
            predictions = pickle.load(f)

        cnn_preds = predictions['{}_CNN'.format(testsite)]
        true =  predictions['{}_truth'.format(testsite)]

        return cnn_preds, true

    def gen_conf_matrix(self, ensemble = True):
        '''

        This will generate confusion matrices for both test sites.


        '''
        class_names = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
        best_models = []

        for model in self.modelnames:
            print(model)
            plot_fname = self.plot_folder + model + 'conf_matrix.png'

            fig, ax = pl.subplots(2,1, tight_layout = {'rect':[0, 0, 1, 0.95]}, sharex = True)

            f1 = np.zeros((2,10))
            for ti,testsite in enumerate(['duck', 'nbn']):

                for run in range(5):

                   cnn_preds, true = self.load_conf_table(model, run, testsite)

                   f1[ti, run] = metrics.f1_score(true, cnn_preds, average='weighted')

                   cnn_preds = [cc.item() for cc in cnn_preds]
                   true = [cc.item() for cc in true]


                   if run == 0:
                    conf_matrix = metrics.confusion_matrix(true, cnn_preds)
                    conf_matrix = np.expand_dims(conf_matrix, axis = 0)

                   if run > 0:
                    conf_matrix_ = metrics.confusion_matrix(true, cnn_preds)
                    conf_matrix_ = np.expand_dims(conf_matrix_, axis = 0)

                    conf_matrix = np.concatenate((conf_matrix, conf_matrix_), axis = 0)

                title = '{} Train on {}'.format(model, self.trainsite)

                if ensemble:
                    self.confusionTable(conf_matrix, class_names, fig, ax[ti], testsite, ensemble = ensemble)


            if not ensemble:
                f1_mean = np.mean(f1, axis = 0) #choose the best performing model and reload the predictions to plot them.
                best_model_index = np.argwhere(f1_mean == np.max(f1_mean))[0][0]
                best_models.append(best_model_index) #return the best performing ensemble member for each model type

                for ti, testsite in enumerate(['duck', 'nbn']):
                    #load the data
                    cnn_preds, true = self.load_conf_table(model, best_model_index, testsite)
                    cnn_preds = [cc.item() for cc in cnn_preds]
                    true = [cc.item() for cc in true]

                    conf_matrix = metrics.confusion_matrix(true, cnn_preds)
                    self.confusionTable(conf_matrix, class_names, fig, ax[ti], testsite, ensemble = ensemble)


            pl.suptitle(title)

            pl.savefig(plot_fname)
            print('Printed Confusion Matrix for {}'.format(model))

        return best_models


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

