#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import os
import cv2
import argparse
import time

import numpy as np
from numpy import random
from pathlib import Path

import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T

from utils.general import apply_classifier, increment_path
from utils.torch_utils import select_device, time_synchronized
from utils.augmentations import letterbox
from utils.inference_util import *

from source.get_act_rec_model import get_action_rec_model
from torchvision.models import resnet34
from torchvision import transforms

from boxmot import StrongSORT, BoTSORT
from ultralytics import YOLOv10, YOLO

#Server RESTApi
import requests
import json
from datetime import datetime

#Server URL
url = 'http://127.0.0.1:5000/api/bbox'

class ObjectDetectionNode:
    def __init__(self, opt):

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
        
        if opt.img_size == 960:
            self.resizeT = T.Resize((544, 960))
        elif opt.img_size == 1280:
            self.resizeT = T.Resize((736, 1280))
        else: 
            self.resizeT = T.Resize((384, 640))
        
        #ROS settings
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber(name = "/camera/image_raw/compressed", data_class = CompressedImage, callback = self.image_callback, queue_size=10)
        
        self.img_result_pub = opt.img_result_pub
        if self.img_result_pub == True:
            self.image_pub = rospy.Publisher("/camera/image_detected/image", Image, queue_size=10)
            
          
    def image_callback(self, msg):
        self.nC = self.nC + 1
        
        # Discard every 2(second) frame from ROS data
        if self.nC % 2 == 0:
            self.YOLO_detection = True 
        else: 
            self.YOLO_detection = False 
            
        self.tracked_EC = []
        
        if self.YOLO_detection:
            t0 = time.time()
            
            np_arr = np.frombuffer(msg.data, np.uint8)  # Decode from byte array
            im0s = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode to OpenCV format
            im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)

            # Padded resize
            #im0s_padd = letterbox(im0s, self.opt.img_size, stride=32)[0]

            if im0s.size != 0:
                # Preprocess Image
                #_img = torch.from_numpy(im0s_padd).to(self.device)
                img = self.resizeT(torch.from_numpy(im0s).to(self.device).permute(2,0,1)) / 255.0
                
                #img = _img / 255.0  # 0 - 255 to 0.0 - 1.0
                #img = img.permute(2,0,1)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                #t_det0 = time_synchronized()
                pred = self.handle_detection(img)
                t_det1 = time_synchronized()         
                
                # Apply Classifier
                if self.opt.classifyEC:
                    if len(pred):
                        pred = apply_classifier(pred, self.modelEC, img, im0s, self.names, self.device)
                
                t_class1 = time_synchronized()
                
                if self.opt.trackTP: ### Tracker only perform Traffic Police class
                    if len(pred):
                        outputs = self.handle_tracking(self.tracker, pred, self.names, im0s, img, self.opt)
                        
                        t_track1 = time_synchronized()
                        
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
                                    
                                    bbox = img[:,:,y1-self.opt.bTH:y2+self.opt.bTH,x1-self.opt.bTH:x2+self.opt.bTH]

                                    resize_tr = transforms.Resize((self.opt.action_img_size, self.opt.action_img_size))
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

                        #elif self.tracked_TP_bbox >= self.opt.num_frame_action_rec and 
                        
                        #No tracked data null -> tracked memory
                        #TDB: Add logic to check when null the memory
                        else: 
                            self.no_tracked_tp = self.no_tracked_tp + 1
                            
                            ### Buffer stays 10 fram without TP, EC detection
                            if self.no_tracked_tp > self.buffer_lifetime:
                                self.tracked_TP = []
                                self.tracked_TP_bbox = []
                                self.tracked_EC = []
                    
                    t_buff1 = time_synchronized()
                    
                    if len(self.tracked_TP) >= self.opt.num_frame_action_rec and self.opt.actionRec:
                        print('Tracked length:', len(self.tracked_TP))
                        
                        _boxImg = torch.cat(self.tracked_TP_bbox, axis=0).unsqueeze(0)
                        
                        if self.opt.keyPointDet:
                            _keyPoint = torch.as_tensor(self.tracked_TP_keypoint).to(self.device).unsqueeze(0) / self.opt.action_img_size
                        
                        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                            if self.opt.keyPointDet:
                                x = [_boxImg, _keyPoint]
                            else: 
                                x = _boxImg
                            actionTP = self.modelRecTP(x, label=None)
                            actionTP = torch.argmax(actionTP, dim=-1).cpu().numpy()
                        
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
                        
                t_ac_rec1 = time_synchronized()
                
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
                
                t_save1 = time_synchronized()
            
            print(f'Done. All elapsed time:({time.time() - t0:.3f}s)\n \
                  \t Detect time: ({t_det1 - t0:.3f}s)\n \
                  \t Class time: ({t_class1 - t_det1:.3f}s)\n \
                  \t Track time: ({t_track1 - t_class1:.3f}s)\n \
                  \t Buff time: ({t_buff1 - t_track1:.3f}s)\n \
                  \t Action rec time: ({t_ac_rec1 - t_buff1:.3f}s)\n \
                  \t Save time: ({t_save1 - t_ac_rec1:.3f}s)\n')  
            
            rospy.loginfo(f'Done. ({time.time() - t0:.3f}s)')  
    
    def load_models(self, opt):
        ### 1. Load detection model 
        model = YOLO(opt.weights)
        #model = YOLOv10(opt.weights, verbose=True)
        #model.to(self.device)
        #model.fuse()
        
        #
        
        # Get names and colors
        #names = model.names
        names = ["Person", "Bike", "Car", "Bus", "Truck", "Traffic Sign", "Traffic Light", "Emergency_Car",  "Traffic_Police"]
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
            # Initialize the StrongSORT tracker
            tracker = BoTSORT(
                #model_weights = Path('./data/weight/tracker/osnet_x1_0_msmt17.pt'),  # which ReID model to use
                model_weights = Path('./data/weight/tracker/osnet_x0_25_msmt17.pt'),  # which ReID model to use
                device = 'cuda:0',
                fp16 = True,
            )
        else: 
            tracker = None
            
        ### 3. Load Action Recognization model 
        # ResnetAttention or ResNetAttentionVisual
        if opt.actionRec:
            if opt.keyPointDet:
                modelRecTP = get_action_rec_model(model_name='ResNetAttention',action_classes=opt.acr_class, num_action_frame=opt.num_frame_action_rec, device=self.device, weightPath=opt.acr_weights)
            else: 
                modelRecTP = get_action_rec_model(model_name='ResNetAttentionVisual',action_classes=opt.acr_class, num_action_frame=opt.num_frame_action_rec, device=self.device, weightPath=opt.acr_weights)
            
            modelRecTP.to(self.device)
            modelRecTP.eval()
            
            #if opt.half:
            #    modelRecTP.half()
        else: 
            modelRecTP = None
        
        ### 4. Load KeyPoint detection model 
        if opt.keyPointDet: 
            modelKPD = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
            modelKPD.eval()
            modelKPD.to(self.device)
            
            #if opt.half:
            #    modelKPD.half()
        else: 
            modelKPD = None
            
        ### 5. Load Emergency Car classification model 
        # resnet34
        if opt.classifyEC:
            emergency_car_class = opt.ec_class
            modelEC = resnet34(pretrained=False, num_classes=emergency_car_class).to('cuda')
            checkpointEC = torch.load(opt.ec_weights, map_location='cuda')
            modelEC.load_state_dict(checkpointEC)
            modelEC.eval()
            
            #if opt.half:
            #    modelEC.half()
            
            #names[9], names[10], names[11], names[12] = 'Police_Car','Fire_Truck','Ambulance','Fire_Other'
            names.append('Police_Car')
            names.append('Fire_Truck')
            names.append('Ambulance')
            names.append('Fire_Other')
            
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        else: 
            modelEC = None
        return model, tracker, modelRecTP, modelKPD, modelEC, colors, names, det_cls_number
            
    def handle_detection(self, img):

        if self.opt.half:
            img = img.half()

        results = self.model(img, conf = self.opt.conf_thres)
        #results = results.half()

        # Convert the detections to the required format: N X (x, y, x, y, conf, cls)
        pred = []
        for result in results:
            for detection in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = detection
                pred.append([x1, y1, x2, y2, conf, int(cls)])
        pred = np.array(pred)
        return pred
        
    def handle_tracking(self, tracker, track_pred, names, im0s, img, opt):
        ### Track only TP, EC 
        track_pred = track_pred[track_pred[:,5]>=7]
        if len(track_pred) > 0:
            outputs = tracker.update(track_pred, im0s)
            if len(outputs):
                outputs[:, [4, 6]] = outputs[:, [6, 4]]
                outputs[:, [4, 5]] = outputs[:, [5, 4]]
                
                ### TDB: -> Track all classes and send only TP, EC 
                ### slower inference speed x10
                ### Send only TP, EC tracked value
                ### outputs = outputs[outputs[:,5]>=(self.opt.tpClassNumber-1)]
                return outputs
        return []
    
    
    def send_server_result_data(self, data, data_type='EC', pred_type=0):
        json_data = convert_data_json_format(data, data_type, pred_type)

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
    parser.add_argument('--weights', nargs='+', type=str, default='./data/weight/detection/yolov10b_keti_tp241120_0_90.pt', help='model.pt path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default='./yolo11l.engine', help='model.pt path(s)')
    parser.add_argument('--ec-weights', nargs='+', type=str, default='./data/weight/classification/resnet34_EC_CL_20241107_Class4.pt', help='model.pt path(s)')
    parser.add_argument('--acr-weights', nargs='+', type=str, default='./data/weight/action_rec/241130_ResNetAttentionVisual_lr0.0005_nf_30_s_2_0.949.pth', help='model.pt path(s)')
        
    parser.add_argument('--source', type=str, default='./data/test/', help='source folder when run from folder')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels) (1280, 960, 640)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--save_img', type=int, default=1, help='save image level 0: no save, 1: save result EC, TP only on img, 2: save all bbox on img')
    parser.add_argument('--img_result_pub', type=bool, default=True, help='Detected result publish ROS')
    
    parser.add_argument('--project', default='runs/wand_gist', help='save results to project/name')
    parser.add_argument('--name', default='ros_result', help='save results to project/name')
    parser.add_argument('--names', default='runs/detect', help='save results tnvidia-oject/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', type=bool, default=False, help='model run half: True/False (Not fully tested yet)')
    
    parser.add_argument('--send_result_server', type=bool, default=True, help='Send result to SERVER by RestAPI')
    parser.add_argument('--classifyEC', type=bool, default=True, help='Classification Emergency Car model: True/False')
    parser.add_argument('--trackTP', type=bool, default=True, help='Track traffic police: True/False')
    parser.add_argument('--actionRec', type=bool, default=True, help='Action Recognization model: True/False')
    
    parser.add_argument('--num_frame_action_rec', type=int, default=30, help='Num of frames for action recognization')
    parser.add_argument('--acr-class', type=int, default=15, help='Action class number, Wand: 7, Hand: 7')
    parser.add_argument('--tpClassNumber', type=int, default=8, help='Traffic Police Class number of Detection')
    parser.add_argument('--keyPointDet', type=bool, default=False, help='Key Point Detection model: True/False')
    parser.add_argument('--bTH', type=int, default=10, help='extend boundary box with threshold(add/sub thr val from bbox x1,x2,y1,y2): 0, 30, 50')
    parser.add_argument('--action-img-size', type=int, default=224, help='action recognization inference size (pixels)')
    parser.add_argument('--tpTRACK_ID', type=int, default=1, help='traffic police track id DEFAULT:1 -> it is update from tracker')
    parser.add_argument('--ec-class', type=int, default=5, help='Emergency classigy class number: Police_Car,Fire_Truck,Ambulance,Fire_Other')
    
    opt = parser.parse_args()
    print(opt)
    
    # Ros Init
    rospy.init_node('ROS_Detection_Node', anonymous=True)
    node = ObjectDetectionNode(opt)
    rospy.spin()