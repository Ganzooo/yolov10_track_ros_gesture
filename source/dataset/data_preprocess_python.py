import os
import glob2
import json
from PIL import Image
import numpy as np
import shutil 

# if not os.path.exists("/dataset/TrafficPoliceData_GIST/train/"): os.mkdir("/dataset/TrafficPoliceData_GIST/train/")

# if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_train/"): os.mkdir("/dataset/TrafficPoliceData_GIST/cropped_train/")
# if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_train2/"): os.mkdir("/dataset/TrafficPoliceData_GIST/cropped_train2/")

# if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_val/"): os.mkdir("/dataset/TrafficPoliceData/cropped_val/")
# if not os.path.exists("/dataset/TrafficPoliceData_GIST/cropped_val2/"): os.mkdir("/dataset/TrafficPoliceData/cropped_val2/")

#if not os.path.exists("/dataset/TrafficPoliceData_GIST/train/"): os.mkdir("/dataset/TrafficPoliceData_GIST/train/")

if not os.path.exists("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_train/"): os.mkdir("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_train/")
if not os.path.exists("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_train2/"): os.mkdir("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/cropped_train2/")

if not os.path.exists("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val/"): os.mkdir("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val/")
if not os.path.exists("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val2/"): os.mkdir("/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val2/")


ACTION_TYPE = 'wand'
#ACTION_TYPE = 'hand' #not implemented
assert ACTION_TYPE == 'wand'
margins_size_w = 60
margins_size_h = 40

# imgPathAll = "/dataset/TrafficPoliceData_GIST/train/"
# jsonPathAll = "/dataset/TrafficPoliceData_GIST/train/"

# newCroppedPath = "/dataset/TrafficPoliceData_GIST/cropped_train/"
# newCroppedPath2 = "/dataset/TrafficPoliceData_GIST/cropped_train2/"

imgPathAll = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/train/"
jsonPathAll = "/dataset/TrafficPoliceData_GIST/val/"

newCroppedPath = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val/"
newCroppedPath2 = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/cropped_val2/"

prepare_file_train = False
if prepare_file_train:
    folder_name = 11
    filePath = f"/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/RawData/image/{folder_name}/"
    filePathList = sorted(glob2.glob(filePath + "/*"))

    for idx, labels in enumerate(filePathList):
        _label = os.path.basename(labels)
        labelPathImg = sorted(glob2.glob(labels + "/*"))
        #for folder in labelPathImg:
        _fName = os.path.basename(_label) + '_' + str(folder_name)
        
        shutil.copytree(labels, os.path.join(imgPathAll,_fName))
        #print(labelPathImg)
    print(filePathList)
    
prepare_file = False
if prepare_file:
    filePath = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/RawData/image/"
    filePathList = sorted(glob2.glob(filePath + "*"))

    for idx, labels in enumerate(filePathList):
        #_label = os.path.basename(labels)
        labelPathImg = sorted(glob2.glob(labels + "/*"))
        for folder in labelPathImg:
            _fName = os.path.basename(folder) + '_' + str(_label)
            
            shutil.copytree(folder, os.path.join(imgPathAll,_fName))
        print(labelPathImg)
    print(filePathList)
    
prepare_file_json = True
if prepare_file_json:
    filePath = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/1_Training/LabelData/json/"
    filePathList = sorted(glob2.glob(filePath + "*"))

    for idx, labels in enumerate(filePathList):
        _label = os.path.basename(labels)
        labelPathImg = sorted(glob2.glob(labels + "/*"))
        for folder in labelPathImg:
            _fName = os.path.basename(folder)[:-5] + '_' + str(_label)
            
            shutil.copyfile(folder, os.path.join(imgPathAll,_fName, _fName+'.json'))
        print(labelPathImg)
    print(filePathList)
    
crop_bbox = False
if crop_bbox:
    trainFolder = sorted(glob2.glob(imgPathAll + "*"))
    for idx, _folder in enumerate(trainFolder):
        train_img = sorted(glob2.glob(_folder + "/*.jpg"))
        folder_name = _folder.split("/")[-1]
        label = folder_name[31:]
        #print("folderName:", folder_name)

        train_json = sorted(glob2.glob(_folder + "/*.json"))
        #train_json_path = path + folder_name + f"/{folder_name}.json"
        #with open(train_json_path, "rb") as f:
        #    js = json.load(f)  
        _newCroppedPath = os.path.join(newCroppedPath, folder_name)
        _newCroppedPath2 = os.path.join(newCroppedPath2, folder_name)
        
        if not os.path.exists(_newCroppedPath):
            os.mkdir(_newCroppedPath)
        if not os.path.exists(_newCroppedPath2):
            os.mkdir(_newCroppedPath2)
        
        for _idx, (imgFile) in enumerate(train_img):
            pil_image = Image.open(imgFile)
            #arr = np.array(pil_image)
            with open(train_json[0], "rb") as f:
                js = json.load(f)  
            image_id = imgFile.split("/")[-1].split('_')[0]
            if js['sequence']['image_id'][_idx] == int(image_id):
                bounding_box = js['sequence']['bounding_box'][_idx]
                bounding_box[0] = int(float(bounding_box[0]))
                bounding_box[1] = int(float(bounding_box[1]))
                bounding_box[2] = int(float(bounding_box[2]))
                bounding_box[3] = int(float(bounding_box[3]))

                
                if len(bounding_box) == 0:
                    break

                pil_image_ = pil_image.crop((bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]))
                filename = os.path.basename(imgFile)
                pil_image_.save(_newCroppedPath + f"/{filename}")

prepare_file_json_scalled = False
if prepare_file_json_scalled:
    filePath = "/dataset_sub/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/LabelData/json/"
    filePathList = sorted(glob2.glob(filePath + "*"))

    for idx, labels in enumerate(filePathList):
        _label = os.path.basename(labels)
        labelPathImg = sorted(glob2.glob(labels + "/*"))
        for folder in labelPathImg:
            _fName = os.path.basename(folder)[:-5] + '_' + str(_label)
            
            with open(folder, "rb") as f:
                js = json.load(f) 
            
            for _idx in range(js['sequence']['image_id']):
                bounding_box = js['sequence']['bounding_box'][_idx]
                bounding_box[0] = int(float(bounding_box[0]))
                bounding_box[1] = int(float(bounding_box[1]))
                # bounding_box[2] = int(float(bounding_box[2]))
                # bounding_box[3] = int(float(bounding_box[3]))
                
                for _2dpox_idx in range(js['sequence']['2d_pos'][_idx]):
                    js['sequence']['2d_pos'][_idx][0] = int(float(js['sequence']['2d_pos'][_idx][0])) - bounding_box[0]
                    js['sequence']['2d_pos'][_idx][1] = int(float(js['sequence']['2d_pos'][_idx][1])) - bounding_box[1]
                    js['sequence']['2d_pos'][_idx][1] = int(js['sequence']['2d_pos'][_idx][1])
    
        with open(os.path.join(_newCroppedPath,_fName, _fName+'.json'), 'w') as f:
            json.dump(js, f)
            #shutil.copyfile(folder, os.path.join(_newCroppedPath,_fName, _fName+'.json'))
        #print(labelPathImg)
    #print(filePathList)

# path = "/dataset/TrafficPoliceData/train/"
# new_path = "/dataset/TrafficPoliceData/cropped_train/"
# new_path2 = "/dataset/TrafficPoliceData/cropped_train2/"
# train = sorted(glob2.glob(path + "*"))
# for i in range(len(train)):
#     if i > -1:
#         train_img = sorted(glob2.glob(train[i] + "/*.jpg"))
#         folder_name = train[i].split("/")[-1]
#         print("folderName:", folder_name)
#         train_json_path = path + folder_name + f"/{folder_name}.json"
#         with open(train_json_path, "rb") as f:
#             js = json.load(f)  
#         new_folder_path = new_path + folder_name
#         new_folder_path2 = new_path2 + folder_name
#         if not os.path.exists(new_folder_path):
#             os.mkdir(new_folder_path)
#         if not os.path.exists(new_folder_path2):
#             os.mkdir(new_folder_path2)
                
#         newTrainJson = new_path + folder_name + f"/{folder_name}.json"
#         newTrainJson2 = new_path2 + folder_name + f"/{folder_name}.json"
#         with open(newTrainJson, "w") as f:
#             json.dump(js, f)
#         with open(newTrainJson2, "w") as f:
#             json.dump(js, f)
        
#         for j in range(len(train_img)):
#             pil_image = Image.open(train_img[j])

#             #arr = np.array(pil_image)
#             bounding_box = js.get('sequence').get('bounding_box')[j]
#             bbox = []
#             for i in range(len(bounding_box)):
#                 if i > 1:
#                     bbox.append(float(bounding_box[i]) + 20)
#                 else:
#                     bbox.append(float(bounding_box[i]) - 20)

#             pil_image_ = pil_image.crop(bbox)
            
            
#             filename = str(j).zfill(3)
#             pil_image_.save(new_folder_path + f"/{filename}.jpg")

#             bbox = []
#             for i in range(len(bounding_box)):
#                 bbox.append(float(bounding_box[i]))

#             pil_image_ = pil_image.crop(bbox)
            
#             filename = str(j).zfill(3)
#             pil_image_.save(new_folder_path2 + f"/{filename}.jpg")
