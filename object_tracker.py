from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchreid.utils import FeatureExtractor

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# class names from COCO
class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]

class Detector:
    def __init__(self, detector_name, class_names):
        self.detector_name = detector_name
        self.class_names = class_names
        if detector_name in ['yolov5n','yolov5s', 'yolov5m', 'yolov5l','yolov5x']:
            self.model = torch.hub.load('ultralytics/yolov5', detector_name)
        else:
            self.model = YoloV3(classes=len(class_names))
            self.model.load_weights('./weights/yolov3.tf')
        
    def run(self, img):
        if self.detector_name in ['yolov5n','yolov5s', 'yolov5m', 'yolov5l','yolov5x']:
            results = self.model(img)
            print('yolov5 prediction:', results.xyxyn)
            results = results.xyxyn[0].cpu().detach().numpy()
            print('result shape:', results.shape)
            indices = np.where(results[:,-1] == 0)
            results = results[indices]
            print('result shape after filter out:', results.shape)
            boxes = results[:,:4]
            scores = results[:,4]
            classes = results[:,-1]
        else:
            # expand dim, then img_in have 4D shape (batch_size, img_shape)
            img_in = tf.expand_dims(img, 0)
            # transform image from Yolo3 utils function
            img_in = transform_images(img_in, 416)

            # Yolo3 prediction
            boxes, scores, classes, nums = yolo.predict(img_in)
            # print('YOLO3 predictions: boxes:', boxes.shape, 'scores:', scores.shape, 'clases:', classes.shape, 'nums:', nums.shape)
            # print('EXAMPLE: boxes:', boxes)
            
            # convert shape to DeepSORT shape
            classes = classes[0]
            scores = scores[0]
            boxes = boxes[0]
        return boxes, scores, classes


class REID:
    def __init__(self, method='mars'):
        self.method = method
        if method=='torchreid':
            self.encoder = FeatureExtractor(
                model_name='osnet_x1_0',
                model_path='deep-person-reid/models/osnet_x1_0_imagenet.pth',
                device='cpu'
            )
        else:
            model_filename = 'model_data/mars-small128.pb'
            self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        
    def __TorchREID_encoder(self, img, boxes):
        imgs = []
        for box in boxes:
            imgs.append(img[boxes[0]:boxes[2], boxes[1]:boxes[3]])
        features = self.encoder(imgs)
        return features
    
    def __default_encoder(self, img, boxes):
        features = self.encoder(img, boxes)
        return features
    
    def run(self, img, boxes):
        if self.method == 'torchreid':
            features = self.__TorchREID_encoder(img, boxes)
        else:
            features = self.__default_encoder(img, boxes)
        return features


# initialize Yolo model
# yolo = YoloV3(classes=len(class_names))
# yolo.load_weights('./weights/yolov3.tf')

# initialize DeepSORT parameters
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

# Encoder for Tracker
# model_filename = 'model_data/mars-small128.pb'
# encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Input video
vid = cv2.VideoCapture('C:/Users/User/Documents/Ba/SMOT/data/video/TUD-Stadtmitte-raw.webm')

# output video
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

# historical trajactory
# from _collections import deque
# pts = [deque(maxlen=30) for _ in range(1000)]

# counter = []

# loading yolov5
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
detector = Detector(detector_name='yolov5n', class_names=class_names)
reid_encoder = REID(method='fastreid')

# Running
while True:
    # imread
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    # convert color due to default opencv is BGR
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
     
    t1 = time.time()
    
    # YOLO5 prediction - START
    # results = model(img_in)
    
    # print('yolov5 prediction:', results.xyxyn)
    
    # boxes = results.xyxyn[0][:,:4].cpu().detach().numpy()
    # scores = results.xyxyn[0][:,4].cpu().detach().numpy()
    # classes = results.xyxyn[0][:,-1].cpu().detach().numpy()
    
    boxes, scores, classes = detector.run(img_in)
    # print('boxes:', boxes, 'scores:', scores, 'classes:', classes) 
    # END
    
    # # YOLOv3 - START:
    # # expand dim, then img_in have 4D shape (batch_size, img_shape)
    # img_in = tf.expand_dims(img_in, 0)
    # print('image input shape:', tf.shape(img_in))
    # # transform image from Yolo3 utils function
    # img_in = transform_images(img_in, 416)

    # # Yolo3 prediction
    # boxes, scores, classes, nums = yolo.predict(img_in)
    # print('YOLO3 predictions: boxes:', boxes.shape, 'scores:', scores.shape, 'clases:', classes.shape, 'nums:', nums.shape)
    # print('EXAMPLE: boxes:', boxes)
    
    # # convert shape to DeepSORT shape
    # classes = classes[0]
    # scores = scores[0]
    # boxes = boxes[0]
    # END
    
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    
    converted_boxes = convert_boxes(img, boxes)
    print('converted_boxes:', len(converted_boxes), converted_boxes)
    # Extract feature for REID
    # features = encoder(img, converted_boxes)
    features = reid_encoder.run(img, converted_boxes)
    print('features shape:', features.shape)

    # Create detection from DeepSORT
    print('before Detection: boxes', len(converted_boxes), 'scores:', scores.shape, 'names:', names.shape, 'features:', features.shape)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores, names, features)]
    print('Detection completed')
    
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    
    # filter out detections
    print('start NMS')
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    print('end NMS')
    
    # update Tracker using Kalman Filter and update Tracker
    print('start tracking process')
    tracker.predict()
    tracker.update(detections)
    print('end update tracking')

    # visualize results
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_count = int(0)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        # visualize historical trajactory
        # center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        # pts[track.track_id].append(center)

        # for j in range(1, len(pts[track.track_id])):
        #     if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
        #         continue
        #     thickness = int(np.sqrt(64/float(j+1))*2)
        #     cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        # height, width, _ = img.shape
        # cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
        # cv2.line(img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)

        
        # Counting
    #     center_y = int(((bbox[1])+(bbox[3]))/2)

    #     if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
    #         if class_name == 'car' or class_name == 'truck':
    #             counter.append(int(track.track_id))
    #             current_count += 1

    # total_count = len(set(counter))
    # cv2.putText(img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    # cv2.putText(img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

    # Display general parameters: FPS
    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    #cv2.resizeWindow('output', vid_height, vid_width)
    cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
    
    #break
vid.release()
out.release()
cv2.destroyAllWindows()