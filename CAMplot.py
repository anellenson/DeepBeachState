import matplotlib.pyplot as pl
import numpy as np
import os

class CAMplot:

    def __init__(self, totalprobs_array, totalpreds, totalvotes_array, testpids, allCAMs, testinps, class_names):
        self.totalprobs_array = totalprobs_array
        self.totalpreds = totalpreds
        self.totalvotes_array = totalvotes_array
        self.testpids = testpids
        self.allCAMs = allCAMs
        self.class_names = class_names
        self.testinps = testinps

    def imshow(inp, mean, std, ax = None, title=None):
        """Imshow for Tensor."""
        ax.imshow(inp)
        if title is not None:
            ax.set_title(title)

    def plot_individual_CAMs(self, out_plot_dir):
        camplotdir = out_plot_dir + '/CAMs/'
        if not os.path.exists(camplotdir):
            os.mkdir(camplotdir)
        imgfolder = '/home/server/pi/homes/aellenso/Research/DeepBeach/images/oblique/test/'
        mean = 0.48
        std = 0.29
        mean = np.array([mean, mean, mean])
        std = np.array([std, std, std])

        fig = pl.figure(1)
        for pid, inp, CAM, probs_CNN, probs_human in zip(self.testpids, self.testinps, self.allCAMs, self.totalprobs_array, self.totalvotes_array):
            inp = inp[0].numpy().transpose((1, 2, 0))
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)

            #image = Image.open(imgfname).convert("L")
            #arr = np.asarray(image)

            ax = pl.subplot(2,2,1)
            ax.axis('off')
            ax.imshow(inp)

            ax = pl.subplot(2,2,2
            ax.axis('off')
            ax.imshow(inp)
            ax.imshow(CAM, alpha = 0.4)

            ax = pl.subplot(2,1,2)
            ax.scatter(np.arange(len(probs_CNN)), probs_CNN, color = 'blue', label = 'CNN')
            ax.plot(np.arange(len(probs_CNN)), probs_CNN,  color = 'blue')

            ax.scatter(np.arange(len(probs_CNN)), probs_human, color = 'black', label = 'Human')
            ax.plot(np.arange(len(probs_CNN)), probs_human,  color = 'black')

            pl.legend()
            ax.set_xlim((0,8))
            ax.set_ylabel('Probability')
            ax.set_xticklabels(self.class_names)
            ax.set_ylim((0,1))
            pl.savefig(camplotdir+'/{}'.format(pid), dpi = 400)


            pl.close()

    def bin_CAMs(self, bins):
        #Sort the CAMs according to how confident it is:
        #REturn the bins of the probabilities and the indices of the images taht belong in that category
        maxprobs = []
        state_and_prob_binned_dict = {}


        for tt in self.totalprobs_array:
            maxprobs.append(np.max(tt))

        #bin the cams according to probability values
        for si, state in enumerate(self.class_names):
            state_and_prob_binned_inds = []
            for bi, bb in enumerate(bins):
                inds = np.where((np.array(maxprobs) >= bb) & (np.array(maxprobs) < bb+0.2))[0]
                state_and_prob_binned_inds.append([ii for ii in inds if self.totalpreds[ii] == si])
            entry = {state:state_and_prob_binned_inds}
            state_and_prob_binned_dict.update(entry)

        return state_and_prob_binned_dict

    def find_and_plot_binned_mean_var(self, state_and_prob_binned_dict, bins, resolution, plotdir):
        #Plots will be each class and level of probability - so 8x4 plots (32 plots)
        for key in list(state_and_prob_binned_dict.keys()):
            fig_1, ax = pl.subplots(4,2, tight_layout = True) #Plot just the mean and the variance for each probability level
            fig_1.set_size_inches(6,9)
            for pi, inds in enumerate(state_and_prob_binned_dict[key]): #Loop through the four probability levels
                try: #This will fail if there are no images in that particular bin
                    binnedCAM = np.zeros((1,resolution,resolution))
                    binnedCAM[0,:,:] = self.allCAMs[inds[0]]

                    for ind in inds:
                        binnedCAM = np.append(binnedCAM,[self.allCAMs[ind]], axis = 0)

                    mean_binned_CAM = np.mean(binnedCAM, axis = 0)
                    variance_binned_CAM = np.var(binnedCAM, axis = 0)

                    ax[pi,0].imshow(mean_binned_CAM)
                    ax[pi,1].imshow(variance_binned_CAM)
                    ax[pi,0].set_ylabel('{} < P < {}'.format(bins[pi], bins[pi] + 0.2))
                except IndexError:
                    continue

            ax[0,0].set_title('Mean')
            ax[0,1].set_title('Variance')
            fig_1.suptitle('{}'.format(key))
            fig_1.savefig(plotdir + 'CAM_mean_and_var_{}.png'.format(key))







