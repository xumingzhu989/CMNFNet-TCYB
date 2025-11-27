#!/usr/bin/python
import os
import glob

rootpath = '../dataset/cod/test/'
datasets  = ['DUTS-TE', 'CAMO_test', 'CHAMELEON_test', 'COD10K_test']

for dataset in datasets:
    dataPath = rootpath + dataset + '/'
    outfile = rootpath + dataset + '_list.txt'

    with open(outfile,"w") as file:
        s = dataPath + 'Image/'
        imgFiles = glob.glob(os.path.join(s +'*.jpg'))
        imgFiles.sort()
        for f in imgFiles:
            file.write(f +"\n")


    
