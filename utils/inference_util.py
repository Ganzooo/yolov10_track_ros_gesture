import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from utils.plots import plot_one_box, plot_one_box_tracked
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_boxes, xyxy2xywh, strip_optimizer, set_logging, increment_path
from collections import deque
from numpy import random
import torch

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
#action_labels = ['Right_to_Left', 'Left_to_Right', 'Front_Stop', 'Rear_Stop','Left_and_Right_Stop', 'Front_and_Rear_Stop']
action_labels = ['Go', 'No_signal', 'Slow', 'Stop_front','Stop_side', 'Turn_left', 'Turn_right']

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        #print(f"Directory '{directory}' created.")
    #else:
        #print(f"Directory '{directory}' already exists.")

def visualize_keypoints(image, keypoints, threshold=0, fname='./test.png'):
    #plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.imshow(image)
    #for keypoint, score in zip(keypoints[0], keypoint_scores[0]):
    for keypoint in keypoints:
        plt.plot(keypoint[0].cpu(), keypoint[1].cpu(), 'ro')
    plt.axis('off')
    plt.savefig(fname)
    plt.close()
    
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 8:  # Bus
        color = (255, 0, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 2  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 2, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    #cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= opt.trailslen)
          #speed_line_queue[id] = [] ##

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img

def plot_tracked_tp_at_img(names, tracked_TP, actionTP, img, im0s, colors, cls_number):
    if cls_number == 6:
        action_labels = ['Right_to_Left', 'Left_to_Right', 'Front_Stop', 'Rear_Stop','Left_and_Right_Stop', 'Front_and_Rear_Stop']
    else:
        action_labels = ['No_signal_hand','Right_to_Left', 'Left_to_Right', 'Front_Stop', 'Rear_Stop','Left_and_Right_Stop', 'Front_and_Rear_Stop','Go', 'Turn_left', 'Turn_right', 'Stop_front','Stop_side', 'No_signal', 'Slow']

    label = f'{names[int(tracked_TP[-1][:,-3])]}:{str(int(tracked_TP[-1][:,-2]))}' + '\n' + 'Action: ' + f'{action_labels[actionTP[0]]}'
    
    _scaled_bbox_xyxy = scale_boxes(img.shape[2:], torch.from_numpy(tracked_TP[-1][:,:4].astype('float32')), im0s.shape).round()
    plot_one_box_tracked(_scaled_bbox_xyxy, im0s, label=label, color=colors[int(tracked_TP[-1][:,-3])], line_thickness=5)
    
def plot_tracked_ec_at_img(names, tracked_EC, img, im0s, colors):
    label = f'{names[int(tracked_EC[-1][:,-3])]}:{str(int(tracked_EC[-1][:,-2]))}'
    
    _scaled_bbox_xyxy = scale_boxes(img.shape[2:], torch.from_numpy(tracked_EC[-1][:,:4].astype('float32')), im0s.shape).round()
    plot_one_box_tracked(_scaled_bbox_xyxy, im0s, label=label, color=colors[int(tracked_EC[-1][:,-1])], line_thickness=5)

def plot_all_boxes_at_img(pred, img, im0s, names, colors):
    try:
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], im0s.shape).round()
    except:
        print('No prediction')
    for i, det in enumerate(pred):  # detections per image
        #det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
        #_scaled_bbox_xyxy = scale_boxes(img.shape[2:], det[:4], im0s.shape).round()
        if len(det):
            *xyxy, conf, cls =  det[:4], det[4], det[5]
            #for *xyxy, conf, cls in reversed(det):
                #label = f'{names[int(cls)]} + conf:{conf}'
            label = f'{names[int(cls)]}'
            _bbox_xyxy = xyxy[0]
            
            plot_one_box(_bbox_xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=5)
                
def save_bbox(tracked_bbox, tracked_keypointDebug, save_dir, opt):
    nC = 0
    
    _path = str(save_dir / str(nC))  # img.jpg
    
    for _idx, boxImg in enumerate(tracked_bbox):
        fname = str(_idx) + '.jpg'
        
        create_directory_if_not_exists(_path)
                                        
        save_path = str(save_dir / str(nC) / fname)
        #boxImgCpu = boxImg.squeeze(0).permute(1,2,0).cpu().numpy()*255
        
        if opt.keyPointDet:
            boxImgCpu = boxImg.squeeze(0).permute(1,2,0).cpu().numpy()
            visualize_keypoints(boxImgCpu,tracked_keypointDebug[_idx], fname=save_path)    
        else: 
            boxImgCpu = boxImg.squeeze(0).permute(1,2,0).cpu().numpy()*255
            cv2.imwrite(save_path, cv2.cvtColor(boxImgCpu.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
    nC = nC + 1
        
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names)) 