import matplotlib.pyplot as pl
import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
import os
import pandas as pd
from sklearn import metrics

class TestResultPlot:

    def __init__(self, modelname, class_names):
        self.modelname = modelname
        self.class_names = class_names

    def calc_errormetrics(self, totalprobs, totalsimplices):
        Pdot = []
        distance = []
        KLdivergence = []
        MAE_per_class = np.zeros((len(totalprobs), len(self.class_names)))
        for di, (CNNprobs, simplex) in enumerate(zip(totalprobs, totalsimplices)):
            Pdot_img = np.dot(CNNprobs, simplex)
            Pdot.append(Pdot_img)
            d_img = np.sqrt(np.sum([(CNN_i - true_i) ** 2 for CNN_i, true_i in zip(CNNprobs, simplex)]))
            distance.append(d_img)
            crossentropy = entropy(simplex, qk=CNNprobs)
            KLdivergence.append(crossentropy)

            d_class = np.abs(CNNprobs - simplex)
            MAE_per_class[di,:] = d_class

        self.Pdot = Pdot
        self.distance = distance
        self.KLdivergence = KLdivergence
        self.MAE = MAE_per_class

        return Pdot, distance, KLdivergence

    def top_and_bottom_quartiles(self, metric):
        sorted_vals = sorted(range(len(metric)), key=lambda k: metric[k])
        fifth_percentile = np.round(0.03*len(sorted_vals))
        low_inds = sorted_vals[:int(fifth_percentile)]
        hi_inds = sorted_vals[-int(fifth_percentile):]
        CAM_inds = np.concatenate((np.array(low_inds), np.array(hi_inds)))

        return CAM_inds

    def plot_metrics(self):
        fig, ax = pl.subplots(3,1, sharex = True)
        ax[0].scatter(np.arange(len(self.Pdot)), self.Pdot)
        ax[0].set_title('Pdot')

        ax[1].scatter(np.arange(len(self.distance)), self.distance)
        ax[1].set_title('Distance')

        ax[2].scatter(np.arange(len(self.KLdivergence)), self.KLdivergence)
        ax[2].set_title('KL Divergence')

        ax[2].set_xlabel('Image Number')
        pl.suptitle(self.modelname)
        pl.savefig('plots/' + self.modelname + '/img_metrics_scatterplot.png')

    def plot_MAE_boxplot(self):
        fig, ax = pl.subplots(1,1)
        ax.boxplot(self.MAE)
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names)

        pl.savefig('plots/' + self.modelname + '/MAEboxplot.png')

    def plot_scatter(self, CNNprobs, humanprobs):

        def plot_line(humanprobs, CNNprobs):
            model = LinearRegression()
            line = model.fit(humanprobs, CNNprobs)
            model_line = line.coef_ * humanprobs + line.intercept_
            score = line.score(humanprobs, CNNprobs)

            return model_line, score

        CNNprobs_cut = CNNprobs.reshape(-1,1)
        humanprobs_cut = humanprobs.reshape(-1,1)
        inds0 = np.where((CNNprobs_cut>1E-2) & (humanprobs_cut>1E-2))[0]
        CNNprobs_cut = CNNprobs_cut[inds0]
        humanprobs_cut = humanprobs_cut[inds0]

        model_line, score = plot_line(humanprobs_cut, CNNprobs_cut)
        fig, ax = pl.subplots(1,1)
        ax.scatter(humanprobs_cut, CNNprobs_cut)
        ax.plot(humanprobs_cut, model_line, 'k')
        ax.text(0.8, 0.9, 'R^2 = {0:.2f}'.format(score))
        ax.set_xlabel('Human')
        ax.set_ylabel('CNN')
        ax.set_title('Total Probs/Preds ' + self.modelname)
        ax.set_xlim((0,1))
        ax.set_ylim((0, 1))
        pl.savefig('plots/' + self.modelname + '_preds_scatter.png')

        fig, axes = pl.subplots(3, 2, tight_layout = {'rect':[0, 0, 1, 0.95]}, sharex= True, sharey =  True)
        fig.set_size_inches(6,6)
        for ii, (CNN, human) in enumerate(zip(CNNprobs.T, humanprobs.T)):
            inds0 = np.where((CNN > 0) & (human > 0.1))[0]
            CNN = CNN[inds0].reshape(-1,1)
            human = human[inds0].reshape(-1, 1)
            model_line, score = plot_line(human, CNN)
            ax = axes.ravel('F')[ii]
            ax.scatter(human, CNN)
            ax.plot(human, model_line, 'k')
            ax.text(0.8, 0.9, 'R^2  = {0:.2f}'.format(score))
            ax.set_title(self.class_names[ii])
            ax.set_xlabel('Human')
            ax.set_ylabel('CNN')
            ax.set_xlim((0,1))
            ax.set_ylim((0, 1))
        fig.suptitle(self.modelname)
        pl.savefig('plots/' + self.modelname + '_preds_scatter_byclass.png')


class ConfResultPlot:

    def __init__(self, class_names):
        self.class_names = class_names

    def return_predictions(self, conf_dt):
        true_labels = []
        pred_labels = []
        for ri, row in enumerate(conf_dt.values):
            true_labels = true_labels + [ri] * int(np.sum(row))
            for ci, column in enumerate(row):
                pred_labels = pred_labels + [ci] * int(column)

        return true_labels, pred_labels


    def plot_conf_dt_mean_and_var(self, modelnames_list, confplotname, all_conf_vals, title):
        #Provide a list of model names and an array of model confidence values
        fig, axes = pl.subplots(len(modelnames_list), 1, tight_layout={'rect':[0, 0, 1, 0.9]})
        fig.set_size_inches(5, 9)
        for mi, (modelname, model_conf_vals) in enumerate(zip(modelnames_list, all_conf_vals)):
            conf_mean = np.mean(model_conf_vals, axis=0)
            conf_std = np.std(model_conf_vals, axis=0)
            ax = axes.ravel('F')[mi]
            cl = ax.pcolor(np.flipud(conf_mean), cmap='Reds')
            for ai, (mean, std) in enumerate(zip(conf_mean.diagonal(), conf_std.diagonal())):
                ax.text(ai + 0.1, 4.5 - ai, '{0:.2f} +/- {1:.2f}'.format(mean, std), fontweight='bold')
            cl.set_clim((0, 1))
            ax.set_xlabel('CNN')
            ax.set_ylabel('Truth')
            ax.set_xticks(np.arange(len(self.class_names)))
            ax.set_xticklabels(self.class_names, rotation=15)
            ax.set_yticks(np.arange(len(self.class_names[::-1])))
            ax.set_yticklabels(self.class_names[::-1])
            ax.set_title(modelname)
        pl.suptitle(title)
        pl.savefig(confplotname)

    def gen_skill_score(self,conf_dt):
        true_labels, pred_labels = self.return_predictions(conf_dt)
        f1 = metrics.f1_score(true_labels, pred_labels, average='weighted')
        corrcoeff = metrics.matthews_corrcoef(true_labels, pred_labels, average='weighted')
        nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)
        return fi, corrcoeff, nmi


    def skill_score_comp_grouped(self, path, modelnames, groupnames, figname):
        #This will plot three groups of models skill scores (f1/correlation coefficient/mutual information')
        #It requires confidence table information which is under the confusion_table_results folder
        barWidth = 0.1
        fig, axes = pl.subplots(3, 1, tight_layout=True, sharex=True)
        fig.set_size_inches([5, 10])
        for mi, metric in enumerate(['f1_score', 'corr_coeff', 'mutual_info']):
            all_score = []
            for model in modelnames:
                err = []
                files = os.listdir(path + model + '/')
                for file in files:
                    conf_dt = pd.read_pickle(path + model + '/' + file)
                    true_labels, pred_labels = self.return_predictions(conf_dt)
                    if metric == 'f1_score':
                        score = metrics.f1_score(true_labels, pred_labels, average='weighted')
                    if metric == 'corr_coeff':
                        score = metrics.matthews_corrcoef(true_labels, pred_labels)
                    if metric == 'mutual_info':
                        score = metrics.normalized_mutual_info_score(true_labels, pred_labels)
                    err.append(score)
                all_score.append(err)
            mean_score = []
            std_score = []
            for scores in all_score:
                mean = np.mean(scores)
                std = np.std(scores)
                mean_score.append(mean)
                std_score.append(std)

            labels = ['orig_ratio_small', 'orig_ratio_big', 'half_height_small', 'half_height_large',
                      'half_width_small', 'half_width_large']
            colors = ['red', 'maroon', 'lightblue', 'navy', 'plum', 'purple']
            b_list = []
            barWidth = 0.1
            ax = axes.ravel('F')[mi]



            for gg in range(len(labels)):
                group1_mean = mean_score[gg*3:gg*3+3]
                group1_std = std_score[gg*3:gg*3+3]

                if gg ==0:
                    r1 = np.arange(len(group1_mean))
                if gg>0:
                    r1 = [x + barWidth for x in r1]

                ax.bar(r1, group1_mean, yerr = group1_std, color=colors[gg], width=barWidth, edgecolor='white', label=labels[gg])
            ax.set_title(metric)

        axes.ravel('F')[2].set_xticks([rr + barWidth for rr in range(len(group1_mean))])
        axes.ravel('F')[2].set_xticklabels(groupnames)
        pl.savefig('plots/' + figname + '.png')
        pl.show()

    def skill_score_comp(self, modelnames, figname):
        #This will plot three groups of models skill scores (f1/correlation coefficient/mutual information')
        #It requires confidence table information which is under the confusion_table_results folder

        fig, axes = pl.subplots(3, 1, tight_layout=True, sharex=True)
        fig.set_size_inches([3, 6])
        for mi, metric in enumerate(['f1_score', 'corr_coeff', 'mutual_info']):
            all_score = []
            for model in modelnames:
                err = []
                files = os.listdir('confusion_table_results/nbn/' + model + '/')
                for file in files:
                    conf_dt = pd.read_pickle('confusion_table_results/nbn/' + model + '/' + file)
                    true_labels, pred_labels = self.return_predictions(conf_dt)
                    if metric == 'f1_score':
                        score = metrics.f1_score(true_labels, pred_labels, average='weighted')
                    if metric == 'corr_coeff':
                        score = metrics.matthews_corrcoef(true_labels, pred_labels)
                    if metric == 'mutual_info':
                        score = metrics.normalized_mutual_info_score(true_labels, pred_labels)
                    err.append(score)
                all_score.append(err)
            mean_score = []
            std_score = []
            for scores in all_score:
                mean = np.mean(scores)
                std = np.std(scores)
                mean_score.append(mean)
                std_score.append(std)

            ax = axes.ravel('F')[mi]
            barWidth = 0.3
            r1 = np.arange(len(mean_score))

            ax.bar(r1, mean_score, yerr = std_score, color='#7f6d5f', width=barWidth, edgecolor='white')
            ax.set_title(metric)

        axes.ravel('F')[2].set_xticks([rr + barWidth for rr in range(len(mean_score))])
        axes.ravel('F')[2].set_xticklabels(['gray', 'rgb'])
        pl.xticks(rotation = 75)
        pl.savefig('plots/' + figname + '.png')
        pl.show()



    def per_class_acc_compare_grouped(self, models_mean, models_std,  groupnames, figname):
        labels = ['h51_w256','h103_w512','h128_w256', 'h256_w512','h256_w128_ds150','h512_w256']
        colors = ['red', 'maroon','lightblue','navy', 'plum', 'purple']
        b_list = []
        barWidth = 0.1
        fig, axes = pl.subplots(3,2, sharex = True, sharey = True)
        fig.set_size_inches(8,8)
        for si, state in enumerate(self.class_names):
            for gg in range(len(labels)):
                ax = axes.ravel('F')[si]
                group1_mean = models_mean[gg*3:gg*3 + 3, si]
                group1_std = models_std[gg*3:gg*3+3, si]

                if gg == 0:
                    x = np.arange(len(group1_mean))
                    r1 = x
                if gg>0:
                    r1 = [xx + barWidth for xx in r1]

                b1 = ax.bar(r1, group1_mean, yerr=group1_std, color=colors[gg], width=barWidth, edgecolor='white', label=labels[gg])
                b_list.append(b1)

                ax.set_title(state)
                ax.set_ylim((0,1))
                ax.set_ylabel('Accuracy (%)')

        ax.set_xticks([rr + barWidth for rr in range(len(group1_mean))])
        ax.set_xticklabels(groupnames)
        axes.ravel('F')[-1].legend(b_list, labels, bbox_to_anchor=[0, 0.5], loc='lower left')
        axes.ravel('F')[-1].axis('off')
        pl.savefig('plots/' + figname + '.png')
        pl.show()

    def per_class_acc_compare(self, models_mean, models_std,  modelnames, figname):
        fig, axes = pl.subplots(3,2, sharex = True, sharey = True)
        fig.set_size_inches(4,4)
        for si, state in enumerate(self.class_names):
            ax = axes.ravel('F')[si]
            state_mean = models_mean[:,si]
            state_std = models_std[:,si]
            barWidth = 0.3
            r1 = np.arange(len(state_mean))

            ax.bar(r1, state_mean, yerr=state_std, color='#2d7f5e', width=barWidth, edgecolor='white')
            ax.set_title(state)
            ax.set_ylim((0,1))
            ax.set_ylabel('Accuracy (%)')

        ax.set_xticks([rr + barWidth for rr in range(len(state_mean))])
        ax.set_xticklabels(['gray', 'rgb', 'removed_shoreline', 'removed_shoreline_histeq'])
        axes.ravel('F')[-1].axis('off')
        pl.savefig('plots/' + figname + '.png')
        pl.show()