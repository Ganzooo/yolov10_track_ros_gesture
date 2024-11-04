import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

#import torch.functional.F as F
import torch.backends.cudnn as cudnn
from numpy import random

from models.common import DetectMultiBackend

from utils.dataloaders import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_boxes, xyxy2xywh, strip_optimizer, set_logging, increment_path
    
from inference_util import *

from utils.torch_utils import select_device, time_synchronized

from collections import deque

import numpy as np
from get_act_rec_model import get_action_rec_model
from torchvision.models import resnet34
from torchvision import transforms


from boxmot import BoTSORT
from ultralytics import YOLOv10

folder_test = True

def send_server_result_data(data):
    pass 

def load_models(opt):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    
    device = select_device(opt.device)
    
    
    ### 1. Load detection model 
    model = YOLOv10(weights)
    
    # Get names and colors
    #names = load_classes(names)
    names = model.names
    det_cls_number = len(names)
    if det_cls_number == 8:
        names = ["Person", "Bike", "Car", "Bus", "Truck", "Traffic Sign", "Traffic Light", "Emergency_Car",  "Traffic_Police"]
        det_cls_number = len(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors[8] = (255,0,255)
    
    if opt.half:
        model.half()  # to FP16
    
    ### 2. Load Tracker model 
    # Deep sort 
    if opt.trackTP:
        # Initialize the tracker
        tracker = BoTSORT(
            model_weights = Path('./data/weight/tracker/osnet_x0_25_msmt17.pt'),  # which ReID model to use
            device = 'cuda:0',
            fp16 = False,
        )

    else: 
        tracker = None
        
    ### 3. Load Action Recognization model 
    # ResnetAttention or ResNetAttentionVisual
    if opt.actionRec:
        if opt.keyPointDet:
            modelRecTP = get_action_rec_model(model_name='ResNetAttention',action_classes=6, num_action_frame=opt.num_frame_action_rec, device=device, weightPath=opt.acr_weights)
        else: 
            modelRecTP = get_action_rec_model(model_name='ResNetAttentionVisual',action_classes=7, num_action_frame=opt.num_frame_action_rec, device=device, weightPath=opt.acr_weights)
        
        modelRecTP.to(device)
        modelRecTP.eval()
        
        if opt.half:
            modelRecTP.half()
    else: 
        modelRecTP = None
    
    ### 4. Load KeyPoint detection model 
    # RCNN Resnet
    if opt.keyPointDet: 
        #modelKPD = keypointrcnn_resnet50_fpn(weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights)
        modelKPD = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        modelKPD.eval()
        modelKPD.to(device)
        
        if opt.half:
            modelKPD.half()
    else: 
        modelKPD = None
        

    ### 5. Load Emergency Car classification model 
    # resnet34
    if opt.classifyEC:
        emergency_car_class = 5
        modelEC = resnet34(pretrained=False, num_classes=emergency_car_class).to('cuda')
        checkpointEC = torch.load(opt.ec_weights, map_location='cuda')
        modelEC.load_state_dict(checkpointEC)
        modelEC.eval()
        if opt.half:
            modelEC.half()

        names = names + ['Ambulance','Fire_Truck','Police_Car', 'EC_Other']
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    else: 
        modelEC = None
        
    return model, tracker, modelRecTP, modelKPD, modelEC, colors, names, det_cls_number
        
def handle_detection(model, img, device, opt):
    
    #with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        # Perform object detection on the frame
    results = model(img, conf = opt.conf_thres)

    # Convert the detections to the required format: N X (x, y, x, y, conf, cls)
    pred = []
    for result in results:
        for detection in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = detection
            pred.append([x1, y1, x2, y2, conf, int(cls)])
    pred = np.array(pred)
        
    # # Inference
    # with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
    #     pred = model(img, augment=opt.augment)[0]

    # # Apply NMS
    # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    return pred
    
def handle_tracking(tracker, pred, names, im0s, img, opt):
    track_pred = pred[pred[:,5]>=7]
    if len(track_pred) > 0:
        outputs = tracker.update(track_pred, im0s)
        if len(outputs):
            outputs[:, [4, 6]] = outputs[:, [6, 4]]
            outputs[:, [4, 5]] = outputs[:, [5, 4]]
            return outputs
    return []
    # xywh_bboxs, confs, oids = [], [], []
    # for det in pred:
    #     if len(det):
    #         for *xyxy, conf, cls in reversed(det):
    #             if names[int(cls)] == 'Traffic_Police':
    #                 x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
    #                 xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
    #                 confs.append([conf.item()])
    #                 oids.append(int(cls))
    # if len(xywh_bboxs) > 0:
    #     xywhs, confss = torch.Tensor(xywh_bboxs), torch.Tensor(confs)
    #     outputs = tracker.update(xywhs, confss, oids, im0s)
    #    return outputs
    # return []

                         

def detect():
    names, source, save_txt, imgsz = opt.names, opt.source, opt.save_txt, opt.img_size

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load models
    model, tracker, modelRecTP, modelKPD, modelEC, colors, names, det_cls_number = load_models(opt)
        
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Run inference
    #if device.type != 'cpu':
    #    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    tracked_TP, tracked_TP_bbox, tracked_TP_keypoint, tracked_TP_keypointDebug, tracked_EC = [], [], [], [], []
    
    nC = 0
    for path, img, im0s, vid_cap in dataset:
        
        # Preprocess Image
        img = torch.from_numpy(img).to(device)
        img = img.half() if opt.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        t0 = time.time()
        pred = handle_detection(model, img, device, opt)
        
        #if len(pred) == 0:
        #    send_server_result_data(pred)
        #    continue
        t1 = time_synchronized
        
        # Apply Classifier
        if opt.classifyEC:
            if len(pred):
                pred = apply_classifier(pred, modelEC, img, im0s, opt.half, names, det_cls_number)

        if opt.trackTP: ### Tracker only perform Traffic Police class
            if len(pred):
                outputs = handle_tracking(tracker, pred, names, im0s, img, opt)
                
                if len(outputs) > 0:
                    for output in outputs:
                        trackID = output[-2]
                        objecCLASS = output[-3]
                        
                        #Object track Traffic Police ID update only when TRACKER NULL
                        if len(tracked_TP) == 0:
                            opt.tpTRACK_ID = trackID
                        
                        if objecCLASS == opt.tpClassNumber and opt.tpTRACK_ID == trackID: # Traffic Police
                            
                            tracked_TP.append(np.expand_dims(output,axis=0))
                            
                            x1, x2, y1, y2  = int(output[0]), int(output[2]), int(output[1]), int(output[3])  
                            
                            bbox = img[:,:,y1-opt.bTH:y2+opt.bTH,x1-opt.bTH:x2+opt.bTH]

                            resize_tr = transforms.Resize((opt.action_img_size, opt.action_img_size))
                            resize_tensor = resize_tr(bbox)
                            
                            if opt.keyPointDet:
                                with torch.no_grad():
                                    keyPointPred = modelKPD(resize_tensor)[0]['keypoints'][0]
                                    pos2list = [item for pair in zip(keyPointPred[:,0], keyPointPred[:,1]) for item in pair] 
    
                                tracked_TP_keypointDebug.append(keyPointPred)
                                tracked_TP_keypoint.append(pos2list)
                            tracked_TP_bbox.append(resize_tensor)
                            if opt.save_img >= 2:
                                save_bbox(tracked_TP_bbox, tracked_TP_keypointDebug, save_dir, opt)
                        
                        #Emergency Car tracked ID check
                        elif objecCLASS > opt.tpClassNumber:
                            tracked_EC.append(output)
                    
                #No tracked data null -> tracked memory
                #TDB: Add logic to check when null the memory
                else: 
                    tracked_TP = []
                    tracked_TP_bbox = []
                    tracked_EC = []
            
            #t4 = time_synchronized()
            #t4a = time_synchronized()
            
            if len(tracked_TP) >= opt.num_frame_action_rec and opt.actionRec:
                print('Tracked length:', len(tracked_TP))
                
                _boxImg = torch.cat(tracked_TP_bbox, axis=0).unsqueeze(0)
                
                if opt.keyPointDet:
                    _keyPoint = torch.as_tensor(tracked_TP_keypoint).to(device).unsqueeze(0) / opt.action_img_size
                t4a = time_synchronized()
                
                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    if opt.keyPointDet:
                        x = [_boxImg, _keyPoint]
                    else: 
                        x = _boxImg
                    actionTP = modelRecTP(x, label=None)
                    actionTP = torch.argmax(actionTP, dim=-1).cpu().numpy()
                print('\nAction',actionTP)
                
                if opt.save_img >= 1:
                    plot_tracked_tp_at_img(names, tracked_TP, actionTP, img, im0s, colors)
                
                #Send result data to SERVER
                #Traffic police data
                if opt.send_result_server:
                    send_server_result_data(tracked_TP[0])
                
                # Remove first frame from MEMORY
                tracked_TP.pop(0)
                tracked_TP_bbox.pop(0)
                if opt.keyPointDet:
                    tracked_TP_keypoint.pop(0)
            else: 
                if opt.save_img >= 1 and len(pred):
                    plot_all_boxes_at_img(pred, img, im0s, names, colors)
        else:
            if opt.save_img >= 1 and len(pred):
                plot_all_boxes_at_img(pred, img, im0s, names, colors)
                
        #t5 = time_synchronized()
        #print(f'{save_path}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (t3a - t3)):.1f}ms) Pre-Track, \
        #      ({(1E3 * (t4 - t3a)):.1f}ms) Track, ({(1E3 * (t4a - t4)):.1f}ms) AcRecPre, ({(1E3 * (t5 - t4a)):.1f}ms) AcRec, ({(1E3 * (t5 - t0a)):.1f}ms) All-time')
            
        #Send result data to SERVER
        #Emergency car data
            if opt.send_result_server:
                send_server_result_data(tracked_EC)
                    
        if opt.save_img >= 1:
            p = Path(path)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0s)
                print(f" The image with the result is saved in: {save_path}")
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='./data/weight/yolov7_x_keti_tp_best_231101.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='./data/weight/detection/yolov10b_keti_tp_0_885_240924.pt', help='model.pt path(s)')
    parser.add_argument('--ec-weights', nargs='+', type=str, default='./data/weight/best_epoch_resnet34_ec_221016.pt', help='model.pt path(s)')
    parser.add_argument('--acr-weights', nargs='+', type=str, default='./data/weight/action_rec/modeltype_ResNetAttentionVisual_image_wand_best_gist.pth', help='model.pt path(s)')
    #parser.add_argument('--acr-weights', nargs='+', type=str, default='./data/weight/action_rec/modeltype_hand_ResNetAttentionVisual_image_best.pth', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./data/TP2/13292_43/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--save_img', type=int, default=2, help='save image level 0: no save, 1: save result img, 2: save each bbox img')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/wand_gist', help='save results to project/name')
    parser.add_argument('--name', default='13292_43', help='save results to project/name')
    parser.add_argument('--names', default='runs/detect', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
    
    parser.add_argument('--num_frame_action_rec', type=int, default=60, help='Num of frames for action recognization')
    parser.add_argument('--actionRec', type=bool, default=True, help='Action Recognization model: True/False')
    parser.add_argument('--tpClassNumber', type=int, default=8, help='Traffic Police Class number of Detection')
    parser.add_argument('--keyPointDet', type=bool, default=False, help='Key Point Detection model: True/False')
    parser.add_argument('--bTH', type=int, default=0, help='extend boundary box with threshold(add/sub thr val from bbox x1,x2,y1,y2): 0, 30, 50')
    parser.add_argument('--classifyEC', type=bool, default=False, help='Classification Emergency Car model: True/False')
    parser.add_argument('--trackTP', type=bool, default=True, help='Track traffic police: True/False')
    parser.add_argument('--half', type=bool, default=False, help='Track traffic police: True/False')
    parser.add_argument('--action-img-size', type=int, default=224, help='action recognization inference size (pixels)')
    parser.add_argument('--tpTRACK_ID', type=int, default=1, help='traffic police track id DEFAULT:1 -> it is update from tracker')
    parser.add_argument('--send_result_server', type=bool, default=False, help='Send Result to SERVER')
    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    opt.exist_ok = True
    
    folder_test = True
    if folder_test == True:
        #src_folder = '/dataset2/dataset2/59_Traffic_Police_Hand_Pattern_Image/01_Data/2_Validation/val_hand'
        src_folder = '/media/tt/data/dataset/Static_test_growth_g/'
        # Loop through all subfolders in the source folder
        for folder in os.listdir(src_folder):
            folder_path = os.path.join(src_folder, folder)
            opt.source = folder_path
            opt.name = folder
            with torch.no_grad():
                detect()
    else: 
        with torch.no_grad():
            detect()