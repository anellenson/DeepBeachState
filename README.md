# DeepBeachState

This repository provides a basic skeleton of how to implement a CNN on Argus data.
The project requires a GPU, which pytorch accesses locally. It runs on python 3.
The github repo includes two dictionaries, where the entries correspond to images and labels for Narrabeen and Duck.
The github repo includes a utils folder, which contains helping scripts that are used by the main scripts.
The run_modules notebook shows how to load/augment the dataset, post-process the results, and plot a Guided-Grad-Cam visualization.  

## Work Flow

Step 0) Download python requirements. See requirements.txt file.

Step 1) Download data.

Step 2) Divide dataset into train/val/test sections and augment images. 
        run_modules.ipynb, first block

Step 3) Train the CNN. In the command line, run:

<pre><code>
python trainResNet.py
</code></pre>
   
The script will output training information that looks like this:

<pre><code>
 Trainfiles length: 4521
 valfiles length: 1496
 For train on nbn, model resnet512_train_on_nbn, Epoch 0/0
 ----------
 train Loss: 1.6569 Acc: 0.2185
 val Loss: 1.5903 Acc: 0.3021
 
 Training complete in 1m 59s
 Best val Acc: 0.302139
 Tested on nbn
 Tested on duck
</code></pre>

Step 4) Post process the prediction results. Evaluate prediction accuracies
        run_modules.ipynb, second block

Step 5) Run guided grad cam on an image:

<pre><code>
python visualize_CNN.py -m 'models/resnet512_train_on_nbn.pth' -i 'images/example_img.jpg' -t 2
</code></pre>

Step 6) Visualize guided grad-cam. 
        run_modules.ipynb, third block
