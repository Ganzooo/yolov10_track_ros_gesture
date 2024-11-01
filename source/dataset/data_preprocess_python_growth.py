import os
import glob2
import json
from PIL import Image
import numpy as np
import shutil 

# imgPathAll = "/dataset/TrafficPoliceData_GIST/train/"
# jsonPathAll = "/dataset/TrafficPoliceData_GIST/train/"

# newCroppedPath = "/dataset/TrafficPoliceData_GIST/cropped_train/"
# newCroppedPath2 = "/dataset/TrafficPoliceData_GIST/cropped_train2/"

newCroppedPath2 = "/dataset/TrafficPoliceData_GIST/cropped_val2/"
newCroppedPath2_g = "/dataset/TrafficPoliceData_GIST/cropped_val2_g/"

newCroppedPath2 = "/dataset/TrafficPoliceData_GIST/cropped_train2/"
newCroppedPath2_g = "/dataset/TrafficPoliceData_GIST/cropped_train2_g/"

prepare_file = True
if prepare_file:
    filePath = newCroppedPath2
    filePathList = sorted(glob2.glob(filePath + "*"))

    for idx, labels in enumerate(filePathList):
        _label = os.path.basename(labels)
        labelPathImg = sorted(glob2.glob(labels + "/*"))
        for folder in labelPathImg:
            for rep in range(4):
                _fName = str(rep) + '_' + os.path.basename(folder)
                newPath = os.path.join(newCroppedPath2_g,_label,_fName)
                os.makedirs(os.path.dirname(newPath), exist_ok=True)
                shutil.copy(folder, newPath)
        print(labelPathImg)
    print(filePathList)

