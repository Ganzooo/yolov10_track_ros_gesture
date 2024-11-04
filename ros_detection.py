#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

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
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)

from collections import deque

import numpy as np
from get_act_rec_model import get_action_rec_model
from torchvision.models import resnet34
from torchvision import transforms



from boxmot import BoTSORT
from ultralytics import YOLOv10

class ObjectDetectionNode:
    def __init__(self, opt):
        # Initialize the YOLO model (you can replace with your custom model)
        #self.model = YOLO("yolov8m.pt")  # Pre-trained YOLOv8 model
        self.bridge = CvBridge()
        print('Init Object Detection Node\n')
    
        self.image_sub = rospy.Subscriber(name = "/camera/image_raw/compressed", data_class = CompressedImage, callback = self.image_callback, queue_size=10)
        
        self.YOLO_detection = True 
        
        self.device = select_device(opt.device)
        self.opt = opt

        # Load models
        self.model, self.tracker, self.modelRecTP, self.modelKPD, self.modelEC, self.colors, self.names, self.det_cls_number = self.load_models(opt)
            
        self.tracked_TP, self.tracked_TP_bbox, self.tracked_TP_keypoint, self.tracked_TP_keypointDebug, self.tracked_EC = [], [], [], [], []
        self.nC = 0
        
        # Directories
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.colors[8] = (255,0,255)
          
    def image_callback(self, msg):
        if self.YOLO_detection:
            t0 = time.time()
            
            np_arr = np.frombuffer(msg.data, np.uint8)  # Decode from byte array
            im0s = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode to OpenCV format
            
            # Padded resize
            im0s_padd = letterbox(im0s, opt.img_size, stride=32)[0]

            if im0s_padd.size != 0:
                self.nC = self.nC + 1
                # Preprocess Image
                _img = torch.from_numpy(im0s_padd).to(self.device)
                
                img = _img / 255.0  # 0 - 255 to 0.0 - 1.0
                img = img.permute(2,0,1)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                t0 = time.time()
                pred = self.handle_detection(img)
                
                t1 = time_synchronized
                
                # Apply Classifier
                if self.opt.classifyEC:
                    if len(pred):
                        pred = self.apply_classifier(pred, self.modelEC, img, im0s, opt.half, self.names, self.det_cls_number)

                if self.opt.trackTP: ### Tracker only perform Traffic Police class
                    if len(pred):
                        outputs = self.handle_tracking(self.tracker, pred, self.names, im0s, img, opt)
                        
                        if len(outputs) > 0:
                            for output in outputs:
                                trackID = output[-2]
                                objecCLASS = output[-3]
                                
                                #Object track Traffic Police ID update only when TRACKER NULL
                                if len(self.tracked_TP) == 0:
                                    opt.tpTRACK_ID = trackID
                                
                                if objecCLASS == self.opt.tpClassNumber and self.opt.tpTRACK_ID == trackID: # Traffic Police
                                    
                                    self.tracked_TP.append(np.expand_dims(output,axis=0))
                                    
                                    x1, x2, y1, y2  = int(output[0]), int(output[2]), int(output[1]), int(output[3])  
                                    
                                    bbox = img[:,:,y1-opt.bTH:y2+opt.bTH,x1-opt.bTH:x2+opt.bTH]

                                    resize_tr = transforms.Resize((opt.action_img_size, opt.action_img_size))
                                    resize_tensor = resize_tr(bbox)
                                    
                                    if self.opt.keyPointDet:
                                        with torch.no_grad():
                                            keyPointPred = self.modelKPD(resize_tensor)[0]['keypoints'][0]
                                            pos2list = [item for pair in zip(keyPointPred[:,0], keyPointPred[:,1]) for item in pair] 
            
                                        self.tracked_TP_keypointDebug.append(keyPointPred)
                                        self.tracked_TP_keypoint.append(pos2list)
                                    self.tracked_TP_bbox.append(resize_tensor)
                                    if self.opt.save_img >= 2:
                                        save_bbox(self.tracked_TP_bbox, self.tracked_TP_keypointDebug, self.save_dir, self.opt)
                                
                                #Emergency Car tracked ID check
                                elif objecCLASS > self.opt.tpClassNumber:
                                    self.tracked_EC.append(output)
                            
                        #No tracked data null -> tracked memory
                        #TDB: Add logic to check when null the memory
                        else: 
                            self.tracked_TP = []
                            self.tracked_TP_bbox = []
                            self.tracked_EC = []
                    
                    #t4 = time_synchronized()
                    #t4a = time_synchronized()
                    
                    if len(self.tracked_TP) >= self.opt.num_frame_action_rec and self.opt.actionRec:
                        print('Tracked length:', len(self.tracked_TP))
                        
                        _boxImg = torch.cat(self.tracked_TP_bbox, axis=0).unsqueeze(0)
                        
                        if self.opt.keyPointDet:
                            _keyPoint = torch.as_tensor(self.tracked_TP_keypoint).to(self.device).unsqueeze(0) / self.opt.action_img_size
                        t4a = time_synchronized()
                        
                        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                            if self.opt.keyPointDet:
                                x = [_boxImg, _keyPoint]
                            else: 
                                x = _boxImg
                            actionTP = self.modelRecTP(x, label=None)
                            actionTP = torch.argmax(actionTP, dim=-1).cpu().numpy()
                        print('\nAction',actionTP)
                        
                        if self.opt.save_img >= 1:
                            plot_tracked_tp_at_img(self.names, self.tracked_TP, actionTP, img, im0s, self.colors, opt.acr_class)
                        
                        #Send result data to SERVER
                        #Traffic police data
                        if self.opt.send_result_server:
                            self.send_server_result_data(self.tracked_TP[0])
                        
                        # Remove first frame from MEMORY
                        self.tracked_TP.pop(0)
                        self.tracked_TP_bbox.pop(0)
                        if self.opt.keyPointDet:
                            self.tracked_TP_keypoint.pop(0)
                    else: 
                        if self.opt.save_img >= 1 and len(pred):
                            plot_all_boxes_at_img(pred, img, im0s, self.names, self.colors)
                else:
                    if self.opt.save_img >= 1 and len(pred):
                        plot_all_boxes_at_img(pred, img, im0s, self.names, self.colors)
                        
                #t5 = time_synchronized()
                #print(f'{save_path}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (t3a - t3)):.1f}ms) Pre-Track, \
                #      ({(1E3 * (t4 - t3a)):.1f}ms) Track, ({(1E3 * (t4a - t4)):.1f}ms) AcRecPre, ({(1E3 * (t5 - t4a)):.1f}ms) AcRec, ({(1E3 * (t5 - t0a)):.1f}ms) All-time')
                    
                #Send result data to SERVER
                #Emergency car data
                    if self.opt.send_result_server:
                        self.send_server_result_data(self.tracked_EC)
                            
                if self.opt.save_img >= 1:
                    # Save the processed image with bounding boxes
                    save_path = os.path.join(self.save_dir, str(self.nC)+'.jpg')
                    self.save_image(save_path, im0s)       
            #print(f'Done. ({time.time() - t0:.3f}s)')  
            rospy.loginfo(f'Done. ({time.time() - t0:.3f}s)')  
    
    def load_models(self, opt):
        ### 1. Load detection model 
        model = YOLOv10(opt.weights)
        
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
                modelRecTP = get_action_rec_model(model_name='ResNetAttention',action_classes=6, num_action_frame=opt.num_frame_action_rec, device=self.device, weightPath=opt.acr_weights)
            else: 
                modelRecTP = get_action_rec_model(model_name='ResNetAttentionVisual',action_classes=opt.acr_class, num_action_frame=opt.num_frame_action_rec, device=self.device, weightPath=opt.acr_weights)
            
            modelRecTP.to(self.device)
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
            modelKPD.to(self.device)
            
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
            
    def handle_detection(self, img):
        
        results = self.model(img, conf = self.opt.conf_thres)

        # Convert the detections to the required format: N X (x, y, x, y, conf, cls)
        pred = []
        for result in results:
            for detection in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = detection
                pred.append([x1, y1, x2, y2, conf, int(cls)])
        pred = np.array(pred)
        return pred
        
    def handle_tracking(self, tracker, pred, names, im0s, img, opt):
        track_pred = pred[pred[:,5]>=7]
        if len(track_pred) > 0:
            outputs = tracker.update(track_pred, im0s)
            if len(outputs):
                outputs[:, [4, 6]] = outputs[:, [6, 4]]
                outputs[:, [4, 5]] = outputs[:, [5, 4]]
                return outputs
        return []
    
    
    def send_server_result_data(self, data):
        pass 

    def save_image(self, file_path, image):
        #file_path = "./runs/ros_result/ros_detected_image.jpg"
        cv2.imwrite(file_path, image)
        rospy.loginfo(f"Image saved at {file_path}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='./data/weight/yolov7_x_keti_tp_best_231101.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='./data/weight/detection/yolov10b_keti_tp_0_885_240924.pt', help='model.pt path(s)')
    parser.add_argument('--ec-weights', nargs='+', type=str, default='./data/weight/best_epoch_resnet34_ec_221016.pt', help='model.pt path(s)')
    #parser.add_argument('--acr-weights', nargs='+', type=str, default='./data/weight/action_rec/modeltype_ResNetAttentionVisual_image_wand_best_gist.pth', help='model.pt path(s)')
    #parser.add_argument('--acr-class', type=int, default=7, help='Action class number, Wand: 7, Hand: 6')
    parser.add_argument('--acr-weights', nargs='+', type=str, default='./data/weight/action_rec/modeltype_hand_ResNetAttentionVisual_image_best.pth', help='model.pt path(s)')
    parser.add_argument('--acr-class', type=int, default=6, help='Action class number, Wand: 7, Hand: 6')
    
    parser.add_argument('--source', type=str, default='./data/TP2/13292_43/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
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
    parser.add_argument('--name', default='ros_result', help='save results to project/name')
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
    
    # Ros Init
    rospy.init_node('object_detection_node', anonymous=True)
    node = ObjectDetectionNode(opt)
    rospy.spin()