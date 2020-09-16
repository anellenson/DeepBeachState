import torch
import os
from torchvision import models, transforms
from PIL import Image
import fnmatch
import pickle

<<<<<<< HEAD
trainsite = 'nbn_duck'
testsite = 'nbn'


img_dirs = {'duck':'/home/aquilla/aellenso/Research/DeepBeach/images/north/match_nbn/', 'nbn':'/home/aquilla/aellenso/Research/DeepBeach/images/Narrabeen_midtide_c5/daytimex_gray_full/'}
basedir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/'
suffices = {'duck':'rect.jpg', 'nbn':'c5.jpg'}
res_height = 512 #height
res_width = 512 #width

labels_pickle = 'labels/{}_labels_dict_five_aug.pickle'.format(testsite)

imgdir = img_dirs[testsite]
suffix = suffices[testsite]

with open(labels_pickle, 'rb') as f:
    labels_dict = pickle.load(f)

for run in range(5):
    modelname = 'resnet512_earlystop_weightdecay_fold{}'.format(run)
    modelpath= '{}/resnet_models/train_on_{}/{}.pth'.format(basedir, trainsite, modelname)

    ##load model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
    nb_classes = len(classes)



    def preprocess(image_path, res_height, res_width):
        transform = transforms.Compose([transforms.Resize((res_height,res_width)), transforms.ToTensor()])
                                           #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            raw_image = transform(image)

        return raw_image, raw_image


    def load_images(test_IDs, res_height, res_width):
        images = []
        raw_images = []

        for ID in test_IDs:
            image, raw_image = preprocess(ID, res_height, res_width)
            images.append(image)
            raw_images.append(raw_image)

        return images, raw_images

    all_imgs = list(labels_dict.keys())
    test_imgs = [aa for aa in all_imgs if suffix in aa]

    truth = [labels_dict[pid] for pid in test_imgs]
    test_IDs = [imgdir + tt for tt in test_imgs]



    images, raw_images = load_images(test_IDs, res_height, res_width)
    images = torch.stack(images).to(device)

    if 'resnet' in modelname:
        model_conv = models.resnet50()
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = torch.nn.Linear(num_ftrs, nb_classes)


    model_conv.load_state_dict(torch.load(modelpath))
    model_conv = model_conv.to(device)
    model_conv.eval()

    predictions = []
    for ii, (image, test_ID) in enumerate(zip(images, test_IDs)):
        image = image.unsqueeze(dim = 0)
        logits = model_conv(image)
        probs = torch.nn.functional.softmax(logits)
        _, prediction= torch.max(logits,1)

        predictions.append(prediction.item())

    predictionary = {'{}_CNN'.format(testsite):predictions, '{}_truth'.format(testsite):truth, 'pids':test_IDs}

    with open('model_output/train_on_{}/{}/predictions_{}.pickle'.format(trainsite,modelname,testsite), 'wb') as f:
        pickle.dump(predictionary, f, protocol = 2)

    print('Finished predictions for fold {}'.format(run))
=======
testsite = 'duck'

for model in ['resnet512_earlystop', 'resnet512_five_aug']:
    for run in range(10):
        modelname = '{}_{}'.format(model, run)
        basedir = '/home/aquilla/aellenso/Research/DeepBeach/python/ResNet/'
        trainsite = 'nbn'
        modelpath= '{}/resnet_models/train_on_{}/{}.pth'.format(basedir, trainsite, modelname)
        out_folder = 'model_output/train_on_{}/{}/'.format(trainsite,  modelname)
        imgdir = '/home/aquilla/aellenso/Research/DeepBeach/images/north/match_nbn/'

        res_height = 512 #height
        res_width = 512 #width

        ##load model

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        classes = ['Ref', 'LTT', 'TBR', 'RBB', 'LBT']
        nb_classes = len(classes)



        def preprocess(image_path, res_height, res_width):
            transform = transforms.Compose([transforms.Resize((res_height,res_width)), transforms.ToTensor()])
                                               #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

            with open(image_path, 'rb') as f:
                image = Image.open(f)
                image = image.convert("RGB")
                raw_image = transform(image)

            return raw_image, raw_image


        def load_images(test_IDs, res_height, res_width):
            images = []
            raw_images = []

            for ID in test_IDs:
                image, raw_image = preprocess(ID, res_height, res_width)
                images.append(image)
                raw_images.append(raw_image)

            return images, raw_images

        augmentations = ['flips', 'gamma', 'rot', 'erase', 'translate', 'vcut', 'streaks']
        # years = ['1986', '1987', '1988']
        # #Find the appropriate images
        # all_imgs = os.listdir(imgdir)
        # test_imgs = []
        # for aa in all_imgs:
        #     year = aa.split('.')[5]
        #     if any([year in aa for year in years]):
        #         if any([sub in aa for sub in augmentations]):
        #             continue
        #         else:
        #             test_imgs.append(aa)
        #

        #filter out augmented images:

        #filter out trainfiles
        with open('../ResNet/labels/{}_daytimex_trainfiles.five_aug.pickle'.format(testsite), 'rb') as f:
            trainfiles = pickle.load(f)

        #test_IDs = [imgdir + tt for tt in test_imgs if tt not in trainfiles]
        test_IDs = [imgdir + tt for tt in trainfiles if not any([sub in tt for sub in augmentations])]

        with open('labels/{}_labels_dict_five_aug.pickle'.format(testsite), 'rb') as f:
            labels_dict = pickle.load(f)


        images, raw_images = load_images(test_IDs, res_height, res_width)
        images = torch.stack(images).to(device)

        if 'resnet' in modelname:
            model_conv = models.resnet50()
            num_ftrs = model_conv.fc.in_features
            model_conv.fc = torch.nn.Linear(num_ftrs, nb_classes)


        model_conv.load_state_dict(torch.load(modelpath))
        model_conv = model_conv.to(device)
        model_conv.eval()

        CNN_site_preds = []
        truth_site = []

        for ii, (image, ID) in enumerate(zip(images, trainfiles)):
            image = image.unsqueeze(dim = 0)
            logits = model_conv(image)
            probs = torch.nn.functional.softmax(logits)
            _, prediction= torch.max(logits,1)

            state = prediction.item()
            label = labels_dict[ID]
            CNN_site_preds.append(state)
            truth_site.append(label)


        predictionary = {'{}_CNN'.format(testsite):CNN_site_preds, '{}_truth'.format(testsite):truth_site}
        with open(out_folder + 'preds_{}.pickle'.format(testsite), 'wb') as f:
            pickle.dump(predictionary, f)

        print('Finished predictions for run {}'.format(run))
>>>>>>> aug_discussion
