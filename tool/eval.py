#!/usr/bin/env python
"""
Demo script showing detections in sample images.
See README.md for installation instructions before running.
"""

import os
import sys
import argparse
from operator import itemgetter
import pprint
import gzip
import pickle

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import cv2
import numpy as np

import matplotlib.pyplot as plt

"""
# 11st 19 classes
CLASSES = ('__background__', # always index 0
           'tshirts', 'shirts', 'blouse', 'knit', 'jacket', 
           'onepiece', 'skirt', 'coat', 'cardigan', 'vest', 
           'pants', 'leggings', 'shoes', 'bag', 'swimwear', 
           'hat', 'panties', 'bra', 'socks')
"""

# 11st 13 clases
CLASSES = ('__background__', # always index 0
           'bag', 'bra', 'jacket_coat', 'onepiece', 'pants', 
           'panty', 'shoes', 'skirt', 'swimwear', 'tshirts_shirts_blouse_hoody', 
           'vest', 'knit_cardigan')


def boxoverlap(gt_box, pred_box):
  x1 = max(gt_box[0], pred_box[0])
  y1 = max(gt_box[1], pred_box[1])
  x2 = min(gt_box[2], pred_box[2])
  y2 = min(gt_box[3], pred_box[3])

  w = x2 - x1 + 1
  h = y2 - y1 + 1
  inter = w * h

  aarea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
  barea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)

  o = float(inter) / float(aarea + barea - inter)

  if(w <= 0): o = 0
  if(h <= 0): o = 0

  return o	


def VOCevaldet(gt_all,
               predicted_all,
               IOU_RATIO=0.7):
  # NOTE: refer to VOCdevkit/VOCcode/VOCevaldet.m
  # url: http://vision.cs.utexas.edu/voc/VOCcode/VOCevaldet.m

  # sort detections by decreasing confidence
  predicted_all = sorted(predicted_all, key=itemgetter(2), reverse=True)

  # get # of true-positive boxes
  npos = 0
  for image_id, gt in gt_all.iteritems():
    npos += gt['num_rois']

  tp = np.zeros(len(predicted_all))
  fp = np.zeros(len(predicted_all))
  for image_idx, predicted_item in enumerate(predicted_all):
    # get predicted result
    image_id = predicted_item[0]
    predicted_cls_idx = int(predicted_item[1])
    confidence = float(predicted_item[2])
    predicted_box = [int(predicted_item[3]), 
                     int(predicted_item[4]), 
                     int(predicted_item[5]), 
                     int(predicted_item[6])]
    # get ground-truth for the image_id
    gt = gt_all[image_id]
    gt_boxes = gt['gt_boxes']
    num_rois = gt['num_rois']
    gt_cls_ids = gt['gt_class_id']

    max_iou = 0
    hit_cls_idx = None
    for box_idx, gt_box in enumerate(gt_boxes):
      iou = boxoverlap(gt_box, predicted_box)
      if max_iou < iou:
        max_iou = iou
        hit_cls_idx = gt_cls_ids[box_idx]

    if max_iou > IOU_RATIO:
      #if confidence > CONF_THRESH and predicted_cls_idx == hit_cls_idx:
      if predicted_cls_idx == hit_cls_idx:
        tp[image_idx] = 1
      else: 
        # false positive (multiple detection)
        fp[image_idx] = 1
    else: 
        # false positive
      fp[image_idx] = 1

  tp = np.cumsum(tp) 
  fp = np.cumsum(fp)

  recall = tp / npos # broadcast
  precision = np.divide(tp, fp+tp) # elewise divice

  average_precision=0.0;
  for threshold in np.arange(0,1,0.1):
    p = 0.0
    # oriignal matlab voc eval code
    #p = np.max(precision(recall>=threshold))
    #if p == None:
    #  p = 0.0
    # NOTE: python converted
    for idx, rec in enumerate(recall):
      if rec > threshold:
        temp_p = precision[idx]
        if p < temp_p:
          p = temp_p

    average_precision += p / 11.0

  print 'average_precision: %f'% (average_precision)

  return average_precision


def vis_detections(im, class_name, dets, CONF_THRESH=0.5):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
  if len(inds) == 0:
    return
  print CLASSES.index(class_name)
  im = im[:, :, (2, 1, 0)]
  fig, ax = plt.subplots(figsize=(12, 12))
  ax.imshow(im, aspect='equal')
  for i in inds :
    bbox = dets[i, :4]
    score = dets[i, -1]

    ax.add_patch(plt.Rectangle((bbox[0], 
                                bbox[1]),
                                bbox[2] - bbox[0],
                                bbox[3] - bbox[1], 
                                fill=False,
                                edgecolor='red', 
                                linewidth=3.5))
    ax.text(bbox[0], 
            bbox[1] - 2,
            '{:s} {:.3f}'.format(class_name, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14, 
            color='white')

  ax.set_title(('{} detections with p({} | box) >= {:.1f}')\
    .format(class_name, class_name, CONF_THRESH), fontsize=14)
  plt.axis('off')
  plt.tight_layout()
  plt.draw()


def detect_for_VOCevaldet(net, 
                          image_filename, 
                          NMS_THRESH=0.3):
  """Detect object classes in an image using pre-computed object proposals."""

  # Load the demo image
  im = cv2.imread(image_filename)

  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(net, im)
  timer.toc()
  #print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, 
  #                                                                  boxes.shape[0])

  # Visualize detections for each class
  objects_box = []
  objects_cls = []
  predicted = []
  for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
             .astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    # NOTE: In evaluation mode, all of the boxes should be considered
    inds = np.where(dets[:, -1] >= 0.0)[0]

    if len(inds) == 0:
      continue	

    for i in inds:
      bbox = dets[i,:]
      # NOTE: prediction result for evaluation
      # Format: list of tuples
      # tuple format: (image_id, cls_ind, confidence, x1, y1, x2, y2)
      predicted.append((image_filename.split('/')[-1][0:-4], 
                        int(cls_ind), 
                        float(bbox[-1]), 
                        int(bbox[0]), 
                        int(bbox[1]), 
                        int(bbox[2]), 
                        int(bbox[3])))
  
  return predicted


def demo(net, image_name, CONF_THRESH=0.67, NMS_THRESH=0.3):
  """Detect object classes in an image using pre-computed object proposals."""

  # Load the demo image
  im = cv2.imread(image_name)

  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(net, im)
  timer.toc()
  print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, 
                                                                    boxes.shape[0])
  # Visualize detections for each class
  for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
             .astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    vis_detections(im, cls, dets, CONF_THRESH=CONF_THRESH)


def parse_args():
  parser = argparse.ArgumentParser(description='Faster R-CNN demo')
  parser.add_argument('--img', dest='image', help='Image to demo')
  parser.add_argument('--thres', dest='thres', help='Threshold of IoU')
  parser.add_argument('--nms_thres', dest='nms_thres', help='Threshold for NMS')
  parser.add_argument('--output', 
                      dest='result_filename', 
                      help='filename for saving gt and detection results')
  parser.add_argument('--cfg', 
                      dest='cfg_file',
                      help='optional config file',
                      default='', 
                      type=str)

  args = parser.parse_args()

  return args

if __name__ == '__main__':
  #import pdb; pdb.set_trace()

  args = parse_args()
  # NOTE: get configuration from yml file or argparse
  if args.cfg_file is not None:
    # conf. from yml
    cfg_from_file(args.cfg_file)
    prototxt = cfg.TEST.PROTOTXT
    caffemodel = cfg.TEST.CAFFE_MODEL
    CONF_THRESH = cfg.TEST.CONF_THRESH
    NMS_THRESH = cfg.TEST.NMS_THRESH
    caffe.set_device(0)
    caffe.set_mode_gpu()
    pprint.pprint(cfg)
  else:
    raise NotImplemented('configurations using argparse does not permitted')

  if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.').format(caffemodel))

  # NOTE: load faster-rcnn model
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)
  print '\n\nLoaded network {:s}'.format(caffemodel)

  # Warmup on a dummy image
  im = 1 * np.ones((300, 500, 3), dtype=np.uint8)
  im_detect(net, im)

  image_filename_prefix = '/storage/product/detection/11st_All/Images'
  if args.image is None :
    metafile = os.path.join('/storage/product/detection/11st_All/Annotations', 
                            'annotations_val.txt')
    gt = {}
    predicted_results = []
    with open(metafile, 'r') as f:
      for line in f:
        if line is None: break
        word = line.strip().split(' ')
        image_filename = word[0]

        im_path = os.path.join(image_filename_prefix, 
                               image_filename + '.jpg')
        try:
          #print('count: %d, %s' % (validation_sample_count, im_path)) 
          predicted = detect_for_VOCevaldet(net, 
                                            im_path, 
                                            NMS_THRESH)
            
        except Exception as err:
          print('ERROR: %s' % im_path)
          continue

        num_rois = int(word[1])
        gt_classes = []
        gt_boxes = []
        for i in range(0, num_rois):
          step = 5*i
          gt_classes.append(int(word[step+2]))
          box = np.array([int(word[step+3]), 
                 int(word[step+4]), 
                 int(word[step+5]), 
                 int(word[step+6])])
          gt_boxes.append(box)

        #predicted_results.extend(predicted)
        for result_box in predicted:
          print('%s %d %f %d %d %d %d' % (result_box[0],
                                          result_box[1],
                                          result_box[2],
                                          result_box[3],
                                          result_box[4],
                                          result_box[5],
                                          result_box[6]))
          #sys.stdout.flush()

        gt[image_filename] = {}
        gt[image_filename]['gt_class_id'] = gt_classes
        gt[image_filename]['gt_boxes'] = gt_boxes
        gt[image_filename]['num_rois'] = int(num_rois)

    with gzip.open(result_filename, 'wb') as f:
      print('Saving gt for all %s' % metafile)
      pickle.dump(gt, f)
        
    for IoU in [0.5, 0.6, 0.7, 0.8, 0.9]:
      mAP = VOCevaldet(gt, 
                       predicted_results, 
                       IOU_RATIO=IoU)
      print 'IoU: %.1f, mAP: %f' % (IoU, mAP)
  else:
    im_path = os.path.join(image_filename_prefix, args.image + '.jpg')
    print im_path
    demo(net, im_path, CONF_THRESH, NMS_THRESH)
    plt.show()
