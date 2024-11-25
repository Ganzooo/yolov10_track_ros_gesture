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



from boxmot import BoTSORT, StrongSORT, DeepOCSORT
from ultralytics import YOLOv10

#Server RESTApi
import requests
import json
from datetime import datetime

#Server URL
url = 'http://127.0.0.1:5000/api/bbox'

class ObjectDetectionNode:
    def __init__(self, opt):
        self.bridge = CvBridge()
        print('Init Object Detection Node\n')
    
        self.image_sub = rospy.Subscriber(name = "/camera/image_raw/compressed", data_class = CompressedImage, callback = self.image_callback, queue_size=10)
        
        self.img_result_pub = True
        if self.img_result_pub == True:
            self.image_pub = rospy.Publisher("/camera/image_detected/image", Image, queue_size=10)
        
        self.YOLO_detection = True 
        
        self.device = select_device(opt.device)
        self.opt = opt

        # Load models
        self.model, self.tracker, self.modelRecTP, self.modelKPD, self.modelEC, self.colors, self.names, self.det_cls_number = self.load_models(opt)
            
        self.tracked_TP, self.tracked_TP_bbox, self.tracked_TP_keypoint, self.tracked_TP_keypointDebug, self.tracked_EC = [], [], [], [], []
        self.nC = 0
        self.no_tracked_tp = 0
        self.buffer_lifetime = 20 #without pred buffer stays with 20 frame
        
        # Directories
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.colors[8] = (255,0,255)
          
    def image_callback(self, msg):
        self.nC = self.nC + 1
        
        if self.nC % 2 == 0:
            self.YOLO_detection = True 
        else: 
            self.YOLO_detection = False 
            
        if self.YOLO_detection:
            t0 = time.time()
            
            np_arr = np.frombuffer(msg.data, np.uint8)  # Decode from byte array
            im0s = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode to OpenCV format
            im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
            
            # Padded resize
            im0s_padd = letterbox(im0s, opt.img_size, stride=32)[0]

            if im0s_padd.size != 0:
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
                        pred = apply_classifier(pred, self.modelEC, img, im0s, self.names, self.device)

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
                                    self.tracked_EC.append(np.expand_dims(output,axis=0))
                     
                        #No tracked data null -> tracked memory
                        #TDB: Add logic to check when null the memory
                        else: 
                            self.no_tracked_tp = self.no_tracked_tp + 1
                            
                            ### Buffer stays 10 fram without TP, EC detection
                            if self.no_tracked_tp > self.buffer_lifetime:
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
                        #print('\nAction',actionTP)
                        
                        if self.opt.save_img >= 1:
                            plot_tracked_tp_at_img(self.names, self.tracked_TP, actionTP, img, im0s, self.colors, opt.acr_class)
                        
                        #Send result data to SERVER
                        #Traffic police data
                        if self.opt.send_result_server:
                            self.send_server_result_data(self.tracked_TP[-1], data_type='TP', pred_type=actionTP)
                        
                        # Remove first frame from MEMORY
                        self.tracked_TP.pop(0)
                        self.tracked_TP_bbox.pop(0)
                        if self.opt.keyPointDet:
                            self.tracked_TP_keypoint.pop(0)
                    
                    # Plot EC tracked bbox
                    elif len(self.tracked_EC) > 0: 
                        if self.opt.save_img >= 1:
                            plot_tracked_ec_at_img(self.names, self.tracked_EC, img, im0s, self.colors)
                            
                        #Send result data to SERVER
                        #Emergency car data
                        if self.opt.send_result_server:
                            self.send_server_result_data(self.tracked_EC, data_type='EC')
                        
                        # Remove first frame from MEMORY
                        self.tracked_EC = []
                        
                    # Plot All bbox
                    else: 
                        if self.opt.save_img >= 2 and len(pred):
                            plot_all_boxes_at_img(pred, img, im0s, self.names, self.colors)
                else:
                    if self.opt.save_img >= 2 and len(pred):
                        plot_all_boxes_at_img(pred, img, im0s, self.names, self.colors)
                        
                
                if self.img_result_pub == True:
                    # Convert OpenCV image back to ROS Image message
                    _img = cv2.cvtColor(im0s, cv2.COLOR_RGB2BGR)
                    detected_img_msg = self.bridge.cv2_to_imgmsg(_img)
                    # Publish the annotated image
                    self.image_pub.publish(detected_img_msg)
                    
                if self.opt.save_img >= 1 and not self.img_result_pub:
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
            # tracker = BoTSORT(
            #     model_weights = Path('./data/weight/tracker/osnet_x0_25_msmt17.pt'),  # which ReID model to use
            #     device = 'cuda:0',
            #     fp16 = False,
            # )
            
            # Initialize the StrongSORT tracker
            # tracker = StrongSORT(
            #     model_weights = Path('./data/weight/tracker/osnet_x0_25_msmt17.pt'),  # which ReID model to use
            #     device = 'cuda:0',
            #     fp16 = False,
            # )
            
            tracker = DeepOCSORT(
                model_weights = Path('./data/weight/tracker/osnet_x0_25_msmt17.pt'),  # which ReID model to use
                device = 'cuda:0',
                fp16 = False,
                asso_func="centroid",
                iou_threshold=0.3  # use this to set the centroid threshold that match your use-case best
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
            
            #ec_names = ['Ambulance','Fire_Truck','Fire_Other','Police_Car','Police_Other']
            #names = names + ['Ambulance','Fire_Truck','Fire_Other','Police_Car','Police_Other']
            #names[9], names[10], names[11], names[12], names[13] = 'Ambulance','Fire_Truck','Fire_Other','Police_Car','Police_Other'
            names[9], names[10], names[11], names[12] = 'Police_Car','Fire_Truck','Ambulance','Fire_Other'
            #ec_cls_num = len(names)
            #for i in range(ec_cls_num):
            #    names[det_cls_number+i].append(ec_names[i])
                
            #names.append(['Ambulance','Fire_Truck','Fire_Other','Police_Car','Police_Other'])
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
        
    def handle_tracking(self, tracker, track_pred, names, im0s, img, opt):
        #track_pred = track_pred[track_pred[:,5]>=7]
        if len(track_pred) > 0:
            outputs = tracker.update(track_pred, im0s)
            if len(outputs):
                outputs[:, [4, 6]] = outputs[:, [6, 4]]
                outputs[:, [4, 5]] = outputs[:, [5, 4]]
                
                ### Send only TP, EC tracked value
                outputs = outputs[outputs[:,5]>=7]
                return outputs
        return []
    
    
    def send_server_result_data(self, data, data_type='EC', pred_type=0):
        
        json_data = convert_data_json_format(data, data_type, pred_type)
        # Convert Python dictionary to JSON string with custom datetime handling
        #json_data = json.dumps(jdata)

        # Set headers for JSON content
        headers = {
            'Content-Type': 'application/json'
        }

        try:
            # Send POST request with serialized JSON data
            response = requests.post(url, data=json_data, headers=headers)

            # Print response status and content from the server
            print(f"Status Code: {response.status_code}")
            print(f"Response Content: {response.text}")
        except:
            print("Failed to establish connection::: check your server IP")

    def save_image(self, file_path, image):
        #file_path = "./runs/ros_result/ros_detected_image.jpg"
        
        cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        rospy.loginfo(f"Image saved at {file_path}")

# Sample JSON data
jdata = {
        "time_stamp": 0,
        "msg_count": 42,
        "num_of_em_vehicles": 0,
        "em_vehicles": [],
        "hand_signal": {
            "type": 0,
            "direction": 0,
            "position": {"u": 0, "v": 0},
            "height": 0,
            "width": 0
        }
    }
    
def convert_data_json_format(data, data_type, pred_type):
    # Load the JSON data into a Python dictionary
    if data_type == 'EC':
        
        # Get the number of emergency vehicles from the JSON data
        num_of_em_vehicles = len(data[0])

        if num_of_em_vehicles > 1:
            pass
        
        # Dynamically generate em_vehicles data based on num_of_em_vehicles
        for i in range(num_of_em_vehicles):
            new_vehicle = {
                'type': data[0][i][5],  # Assign a unique type for each vehicle (you can customize this)
                'position': {
                    'u': round(data[0][i][0],2),  # Example dynamic u position
                    'v': round(data[0][i][1],2)   # Example dynamic v position
                },
                'height': round(data[0][i][2]-data[0][i][0],2),   # Example dynamic height
                'width': round(data[0][i][3]-data[0][i][1],2)     # Example dynamic width
            }
            
            # Append each new vehicle to the em_vehicles list
            jdata['em_vehicles'].append(new_vehicle)
    elif data_type == 'TP':
        # Modify values inside the 'hand_signal' object
        jdata['hand_signal']['type'] = 0
        jdata['hand_signal']['direction'] = int(pred_type)
        jdata['hand_signal']['position']['u'] = round(data[0][0],2)
        jdata['hand_signal']['position']['v'] = round(data[0][1],2)
        jdata['hand_signal']['height'] = round(data[0][2]-data[0][0],2)
        jdata['hand_signal']['width'] = round(data[0][3]-data[0][1],2)
    else: 
        print('Need to choose proper data type!!!!')

    jdata['time_stamp'] = str(datetime.now())
    jdata['msg_count'] = len(jdata)
    
    # Convert the updated dictionary back to a JSON string (for output or further use)
    updated_json = json.dumps(jdata, indent=4)
    return updated_json
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='./data/weight/yolov7_x_keti_tp_best_231101.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='./data/weight/detection/yolov10b_keti_tp241120_0_90.pt', help='model.pt path(s)')
    parser.add_argument('--ec-weights', nargs='+', type=str, default='./data/weight/classification/resnet34_EC_CL_20241107_Class4.pt', help='model.pt path(s)')
    parser.add_argument('--acr-weights', nargs='+', type=str, default='./data/weight/action_rec/modeltype_ResNetAttentionVisual_CLS14_hand_wand_0_92_241112.pth', help='model.pt path(s)')
    parser.add_argument('--acr-class', type=int, default=15, help='Action class number, Wand: 7, Hand: 7')
    #parser.add_argument('--acr-weights', nargs='+', type=str, default='./data/weight/action_rec/modeltype_hand_ResNetAttentionVisual_image_best.pth', help='model.pt path(s)')
    #parser.add_argument('--acr-class', type=int, default=6, help='Action class number, Wand: 7, Hand: 6')
    
    parser.add_argument('--source', type=str, default='./data/TP2/13292_43/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--save_img', type=int, default=2, help='save image level 0: no save, 1: save result EC, TP only on img, 2: save all bbox on img')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/wand_gist', help='save results to project/name')
    parser.add_argument('--name', default='ros_result', help='save results to project/name')
    parser.add_argument('--names', default='runs/detect', help='save results tnvidia-oject/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
    
    parser.add_argument('--num_frame_action_rec', type=int, default=60, help='Num of frames for action recognization')
    parser.add_argument('--actionRec', type=bool, default=True, help='Action Recognization model: True/False')
    parser.add_argument('--tpClassNumber', type=int, default=8, help='Traffic Police Class number of Detection')
    parser.add_argument('--keyPointDet', type=bool, default=False, help='Key Point Detection model: True/False')
    parser.add_argument('--bTH', type=int, default=10, help='extend boundary box with threshold(add/sub thr val from bbox x1,x2,y1,y2): 0, 30, 50')
    parser.add_argument('--classifyEC', type=bool, default=True, help='Classification Emergency Car model: True/False')
    parser.add_argument('--trackTP', type=bool, default=True, help='Track traffic police: True/False')
    parser.add_argument('--half', type=bool, default=False, help='Track traffic police: True/False')
    parser.add_argument('--action-img-size', type=int, default=224, help='action recognization inference size (pixels)')
    parser.add_argument('--tpTRACK_ID', type=int, default=1, help='traffic police track id DEFAULT:1 -> it is update from tracker')
    parser.add_argument('--send_result_server', type=bool, default=True, help='Send Result to SERVER')
    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    opt.exist_ok = True
    
    # Ros Init
    rospy.init_node('object_detection_node', anonymous=True)
    node = ObjectDetectionNode(opt)
    rospy.spin()