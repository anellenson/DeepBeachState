import matplotlib.pyplot as pl
import pickle
from PIL import Image
import matplotlib.pyplot as pl
import plotTools as pt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics

###########Compare models



############Plot gradcam
testsite = 'nbn'
fig, ax = pl.subplots(5, topk + 1, tight_layout = {'rect':[0,0, 1, 0.95]}, figsize = [10,15])
fig.subplots_adjust(0,0,0.9,1)
pl.suptitle('Saliency Maps: Tested at {}'.format(testsite), fontsize = 20)
for j, (image, img_ID) in enumerate(zip(images, test_IDs)):

    image = image.unsqueeze(dim = 0)
    ID = img_ID.split('/')[-1]
    ID = ID.split('.')[0]
    if testsite == 'nbn':
        ID = ID.split('_')[1]
    ax[j,0].imshow(image.squeeze().cpu().numpy().transpose(1,2,0))
    ax[j,0].axis('off')


