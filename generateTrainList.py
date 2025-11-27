#!/usr/bin/python
import os
import glob

rootpath = './datasets/'
# datasets = ['ORS-4199-TR','ORS-4199-TE']
# datasets = ['ORS4199_train16000']
# datasets = ['COD10K_CAMO_CombinedTrainingDataset']
# datasets = ['EORSSD11200_ORS16000_combinedTrain']
# datasets = ['EORSSD1400_ORS2000_combinedTrain']
# datasets = ['ORSSD_train4800','ORSSD-TE','EORSSD_train11200','EORSSD-TE','ORS4199_train16000','ORS-4199-TE']
datasets = ['ORS4199_train16000','ORS-4199-TE']
for dataset in datasets:
    dataPath = rootpath + dataset + '/'
    outfile = rootpath + dataset + '_list.txt'

    with open(outfile,"w") as file:
        p_img = dataPath + 'Image/'
        imgFiles = glob.glob(os.path.abspath(os.path.join(p_img, '*.jpg')))
        imgFiles.sort()
        p_gt = dataPath + 'GT/'
        gtFiles = glob.glob(os.path.abspath(os.path.join(p_gt, '*.png')))
        gtFiles.sort()

        Num_files = len(imgFiles)
        for i in range(Num_files):
            file.write(imgFiles[i].replace('\\','/') + ' ' + gtFiles[i].replace('\\','/') + "\n")

