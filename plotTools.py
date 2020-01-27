import matplotlib
from matplotlib import pyplot as plt
import torch
import numpy as np



def trainInfo_conf_dt(conf_dt, class_names, val_acc, train_acc, val_loss, train_loss, plot_fname, title):
    confusion_matrix = torch.Tensor(conf_dt.values)
    class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    confusion_matrix = confusion_matrix.numpy()
    confusion_text = []
    for row in confusion_matrix:
        confusion_text.append(['%1d' %x for x in row])
    for ri,rr in enumerate(class_acc):
        confusion_text[ri].append('{0:1.1f}'.format(rr*100))

    class_names_withacc = class_names + ['Acc']
    fig = plt.figure()
    plt.clf()
    plt.subplot(221)
    plt.plot(np.arange(0,len(val_acc)), val_acc, color= 'purple', label = 'val')
    plt.plot(np.arange(0,len(val_acc)), train_acc, color = 'orange', label = 'train')
    plt.legend()
    plt.xlabel('Epoch')
    plt.title('Accuracy')

    plt.subplot(222)
    plt.plot(np.arange(0,len(val_loss)), val_loss, color = 'purple', label = 'val')
    plt.plot(np.arange(0,len(train_loss)), train_loss, color = 'orange', label = 'train')
    plt.xlabel('Epoch')
    plt.title('Loss')

    plt.subplot(212)
    plt.table(cellText = confusion_text, rowLabels = class_names_withacc[:-1], colLabels = class_names_withacc, loc = 'center' )
    plt.axis('off')

    plt.suptitle(title)

    plt.savefig(plot_fname, dpi = 600)


def trainInfo(val_acc, train_acc, val_loss, train_loss, plot_fname, title):

    fig = plt.figure()
    plt.clf()
    plt.subplot(121)
    plt.plot(np.arange(0,len(val_acc)), val_acc, color= 'purple', label = 'val')
    plt.plot(np.arange(0,len(val_acc)), train_acc, color = 'orange', label = 'train')
    plt.legend()
    plt.xlabel('Epoch')
    plt.title('Accuracy')

    plt.subplot(122)
    plt.plot(np.arange(0,len(val_loss)), val_loss, color = 'purple', label = 'val')
    plt.plot(np.arange(0,len(train_loss)), train_loss, color = 'orange', label = 'train')
    plt.xlabel('Epoch')
    plt.title('Loss')

    plt.suptitle(title)

    plt.savefig(plot_fname, dpi = 600)




def confusionTable(conf_dt, class_names, plot_fname, title):
    confusion_matrix = torch.Tensor(conf_dt.values)
    class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
    confusion_matrix = confusion_matrix.numpy()
    confusion_text = []
    for row in confusion_matrix:
        confusion_text.append(['%1d' %x for x in row])
    for ri,rr in enumerate(class_acc):
        confusion_text[ri].append('{0:1.1f}'.format(rr*100))

    class_names.append('Acc')
    fig = plt.figure()
    plt.table(cellText = confusion_text, rowLabels = class_names[:-1], colLabels = class_names, loc = 'center' )
    plt.axis('off')
    plt.suptitle(title)
    plt.savefig(plot_fname, dpi = 600)


def confPercent(confpercent_dt, class_names, plot_fname, title):
    confusion_matrix = torch.Tensor(confpercent_dt.values)
    confusion_matrix = confusion_matrix.numpy()
    confusion_text = []
    for row in confusion_matrix:
        confusion_text.append(['%1.2f' %x for x in row])

    fig = plt.figure()
    plt.subplot(211)
    plt.table(cellText = confusion_text, rowLabels = class_names, colLabels = class_names, loc = 'center' )

    plt.subplot(212)
    plt.pcolor(np.flipud(confusion_matrix))
    plt.colorbar()


    plt.axis('off')
    plt.suptitle(title)
    plt.savefig(plot_fname,dpi = 600)

def poster_conf_table(confusion):
    #If you provide a confusion table array, this will plot the confusion table with the numbers labelled within the cells
    fig, ax = pl.subplots(1,1)
    fig.set_size_inches(7,4)
    a1 = ax.pcolor(confusion, cmap = 'Greys')
    for row in np.arange(len(confusion)):
        for col in np.arange(len(confusion)):
            if confusion[row,col] >= 30:
                ax.text(row +0.35, col+0.65, str(int(confusion[row, col])), fontsize = 20, fontweight = 'bold', color = 'white')
            if confusion[row,col] < 30:
                ax.text(row+0.35, col+0.65, str(int(confusion[row, col])), fontsize = 20, fontweight = 'bold')
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis
    ax.yaxis.tick_left()
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
