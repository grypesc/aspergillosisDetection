import copy
import cv2
import os
import pydicom

dataDir = './data/noFungus'
outDir = './data/noLungs'

for dirpath, subdirs, files in os.walk(dataDir):
    if dirpath.find("Dose") != -1: print("Warning, there are redundant folders in data directory")
    pathToWrite = dirpath.replace("noFungus", "noLungs")
    if not os.path.exists(pathToWrite): os.mkdir(pathToWrite)
    # for file in files:
        # ds = pydicom.read_file(dirpath + "/" + file)  # read dicom image
        # img = ds.pixel_array  # get image array
    #     cv2.imwrite(pathToWrite + "/" + file + ".png", img)  # write png image
