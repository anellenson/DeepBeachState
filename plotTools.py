from matplotlib import pyplot as pl
import numpy as np
from sklearn import metrics
import pandas as pd
import pickle


class skillComp():

    def __init__(self, modelnames, plot_folder, out_folder):
        self.modelnames = modelnames
        self.plot_folder = plot_folder
        self.out_folder = out_folder


    def gen_skill_score(self, true_labels, pred_labels):

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
            with open(self.out_folder +'{}/cnn_preds.pickle'.format(model), 'rb') as f:
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


    def confusionTable(self, confusion_matrix, class_names, title, ax, cmap):
        class_acc = confusion_matrix.diagonal()/np.sum(confusion_matrix, axis =1)

        ax.pcolor(confusion_matrix, cmap = cmap)
        for row in np.arange(len(confusion_matrix)):
            for col in np.arange(len(confusion_matrix)):
                if confusion_matrix[row, col] >= 30:
                    ax.text(col +0.35, row+0.65, str(int(confusion_matrix[row, col])), fontsize = 20, fontweight = 'bold', color = 'white')
                if confusion_matrix[row,col] < 30:
                    ax.text(col+0.35, row+0.65, str(int(confusion_matrix[row, col])), fontsize = 20, fontweight = 'bold')
        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.yaxis.tick_left()
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_title(title)



    def gen_conf_matrix(self):
        '''

        This will generate confusion matrices for both test sites.


        '''
        class_names = ['Ref', 'LTT-B', 'TBR-CD', 'RBB-E', 'LBT-FG']

        for model in self.modelnames:
            plot_fname = self.plot_folder + model + 'conf_matrix.png'

            with open(self.out_folder + '{}/cnn_preds.pickle'.format(model), 'rb') as f:
                predictions = pickle.load(f)

            fig, ax = pl.subplots(2,1, tight_layout = {'rect':[0, 0, 1, 0.95]}, sharex = True)
            for ti,testsite in enumerate(['duck', 'nbn']):
                if 'duck' in testsite:
                    cmap = "Blues"

                if 'nbn' in testsite:
                    cmap = "Reds"


                cnn_preds = predictions['{}_CNN'.format(testsite)]
                true =  predictions['{}_truth'.format(testsite)]

                cnn_preds = [cc.item() for cc in cnn_preds]
                true = [cc.item() for cc in true]

                conf_matrix = metrics.confusion_matrix(true, cnn_preds)

                title = 'test on {}'.format(testsite)
                self.confusionTable(conf_matrix, class_names, title, ax[ti], cmap)

            pl.suptitle(model)

            pl.savefig(plot_fname)
            print('Printed Confusion Matrix for {}'.format(model))


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

