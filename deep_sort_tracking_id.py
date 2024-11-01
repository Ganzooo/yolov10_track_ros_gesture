import argparse
import time
from pathlib import Path

import cv2
import torch
#import torch.functional.F as F
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_one_box_tracked
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

import numpy as np

from get_act_rec_model import get_action_rec_model

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}


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
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names)) 
def detect(save_img=False):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    if opt.trackTP:
        # initialize deepsort
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)
    half = True
    if half:
        model.half()  # to FP16

    # Track Traffic Police bbox
    #trackTP = True
    
    # Recognize Traffic Police action 
    #recTP = True
    #num_action_frame = 64
    if opt.actionRec:
        modelRecTP = get_action_rec_model(num_action_frame=opt.num_frame_action_rec)
        modelRecTP.to(device)
        modelRecTP.half()
    
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  #initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    #names = load_classes(names)
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors[8] = (255,0,255)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    tracked_output = []
    tracked_bbox = []
    stack_torch = torch.zeros(3,224,224).to('cuda')
    for path, img, im0s, vid_cap in dataset:
        t0a = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        t3a = time_synchronized()
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        if opt.trackTP: ### Tracker only perform Traffic Police class
            xywh_bboxs = []
            confs = []
            oids = []
            im0 = im0s
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        if names[int(cls)] == 'Traffic_Police':
                            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                            xywh_bboxs.append(xywh_obj)
                            confs.append([conf.item()])
                            oids.append(int(cls))
        
                            xywhs = torch.Tensor(xywh_bboxs)
                            confss = torch.Tensor(confs)
                            
                            outputs = deepsort.update(xywhs, confss, oids, im0s)
                            if len(outputs) > 0:
                                bbox_xyxy = outputs[:, :4]
                                identities = outputs[:, -2]
                                object_id = outputs[:, -1]

                                #label = 'ID:' + f'{identities[0]}{names[int(cls)]} {conf:.2f}'
                                #plot_one_box_tracked(bbox_xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                                
                                if object_id == 8: # Traffic Police
                                    tracked_output.append(outputs)
                                    
                                    x1, x2, y1, y2  = outputs[0][0], outputs[0][2], outputs[0][1], outputs[0][3]  
                                    #bbox = im0[y1:y2,x1:x2,:]
                                    bbox = img[:,:,y1:y2,x1:x2]
                                    
                                    ## Test for cropped box --> TMP
                                    #p = Path(path)  # to Path
                                    #save_path = str(save_dir / p.name)  # img.jpg
                                    #_bbox = bbox.detach().squeeze(0).permute(1,2,0).cpu().float().numpy()*255
                                    #cv2.imwrite(save_path, cv2.cvtColor(_bbox, cv2.COLOR_BGR2RGB))
        
                                    t3a = time_synchronized()
                                    
                                    resize_tensor = torch.functional.F.interpolate(bbox, size=(224, 224), mode='bilinear')
                                    tracked_bbox.append(resize_tensor)
                                    # resize_bbox = cv2.resize(bbox,(224,224))#.transpose(2,0,1)
                                    
                                    
                                    
                                    # resize_bbox = resize_bbox.transpose(2,0,1) / 255.
                                    # tracked_bbox.append(resize_bbox)
                                    
                                    # resize_tensor = torch.from_numpy(bbox).to('cuda').permute(2, 0, 1).unsqueeze(0).half()
                                    # resize_tensor = torch.functional.F.interpolate(resize_tensor, size=(224, 224), mode='bilinear') /255.
                                    # tracked_bbox.append(resize_tensor)
                            else: 
                                tracked_output = []
                                tracked_bbox = []
            t4 = time_synchronized()
            t4a = time_synchronized()
            
            if len(tracked_output) >= opt.num_frame_action_rec and opt.actionRec:
                print('Tracked length:', len(tracked_output))
                #_boxImg = torch.from_numpy(np.expand_dims(tracked_bbox,axis=0))
                #_boxImg = _boxImg.type(torch.FloatTensor).to('cuda').half()
                
                _boxImg = torch.cat(tracked_bbox, axis=0).unsqueeze(0)
                t4a = time_synchronized()
                
                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    actionTP = modelRecTP(_boxImg, label=None)
                    actionTP = torch.argmax(actionTP, dim=1).cpu().numpy()
                print('\nAction',actionTP)
                
                label = 'ID:' + f'{tracked_output[-1][:,-2]}{names[int(tracked_output[-1][:,-1])]} {conf:.2f}' + 'Acc:' + f'{actionTP[0]}'
                _bbox_xyxy = tracked_output[-1][:,:4]
                plot_one_box_tracked(_bbox_xyxy, im0s, label=label, color=colors[int(tracked_output[-1][:,-1])], line_thickness=2)
                
                tracked_output.pop(0)
                tracked_bbox.pop(0)
        t5 = time_synchronized()
                
        p = Path(path)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg    
        print(f'{save_path}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (t3a - t3)):.1f}ms) Pre-Track, \
              ({(1E3 * (t4 - t3a)):.1f}ms) Track, ({(1E3 * (t4a - t4)):.1f}ms) AcRecPre, ({(1E3 * (t5 - t4a)):.1f}ms) AcRec, ({(1E3 * (t5 - t0a)):.1f}ms) All-time')    
            
        if save_img:
            p = Path(path)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0s)
                print(f" The image with the result is saved in: {save_path}")
          
###   
        # # Process detections
        # for i, det in enumerate(pred):  # detections per image
        #     if webcam:  # batch_size >= 1
        #         p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        #     else:
        #         p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        #     p = Path(p)  # to Path
        #     save_path = str(save_dir / p.name)  # img.jpg
        #     txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        #         # Print results
        #         for c in det[:, -1].unique():
        #             n = (det[:, -1] == c).sum()  # detections per class
        #             s += '%g %ss, ' % (n, names[int(c)])  # add to string
        #         xywh_bboxs = []
        #         confs = []
        #         oids = []
        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
        #             xywh_obj = [x_c, y_c, bbox_w, bbox_h]
        #             xywh_bboxs.append(xywh_obj)
        #             confs.append([conf.item()])
        #             oids.append(int(cls))
        #             if save_txt:  # Write to file
        #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        #                 line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
        #                 with open(txt_path + '.txt', 'a') as f:
        #                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

        #             #if save_img or view_img:  # Add bbox to image
        #                 #label = f'{names[int(cls)]} {conf:.2f}'
        #                 #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
        #         xywhs = torch.Tensor(xywh_bboxs)
        #         confss = torch.Tensor(confs)
                
        #         outputs = deepsort.update(xywhs, confss, oids, im0)
        #         if len(outputs) > 0:
        #             bbox_xyxy = outputs[:, :4]
        #             identities = outputs[:, -2]
        #             object_id = outputs[:, -1]

        #             draw_boxes(im0, bbox_xyxy, names, object_id,identities)
            # Print time (inference + NMS)


    #         # Stream results
    #         if view_img:
    #             cv2.imshow(str(p), im0s)
    #             cv2.waitKey(1)  # 1 millisecond

    #         # Save results (image with detections)
    #         if save_img:
    #             if dataset.mode == 'image':
    #                 cv2.imwrite(save_path, im0s)
    #                 print(f" The image with the result is saved in: {save_path}")
    #             else:  # 'video' or 'stream'
    #                 if vid_path != save_path:  # new video
    #                     vid_path = save_path
    #                     if isinstance(vid_writer, cv2.VideoWriter):
    #                         vid_writer.release()  # release previous video writer
    #                     if vid_cap:  # video
    #                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                     else:  # stream
    #                         fps, w, h = 30, im0s.shape[1], im0s.shape[0]
    #                         save_path += '.mp4'
    #                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #                 vid_writer.write(im0s)

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./data/weight/yolov7_x_keti_tp_best_231101.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./data/11_turn2/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--names', default='runs/detect', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
    
    parser.add_argument('--num_frame_action_rec', type=int, default=64, help='Num of frames for action recognization')
    parser.add_argument('--actionRec', type=bool, default=True, help='Action Recognization model: True/False')
    parser.add_argument('--trackTP', type=bool, default=True, help='Track traffic police: True/False')
    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    opt.exist_ok = True
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
