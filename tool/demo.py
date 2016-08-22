#!/usr/bin/env python
"""
Demo script showing detections in sample images.
See README.md for installation instructions before running.
"""

import os
import sys
import argparse
import numpy as np

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt

import caffe
import cv2

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
"""

NETS = {'vgg16': 
        ('VGG16', '/usrdata2/workspace/faster_rcnn_online/output/eleven_all_vgg16_scale_jitter/eleven_all_train/eleven_all_vgg16_faster_rcnn_anneal_stepsize200000_iter_1600000.caffemodel'),
        'res50': 
        ('RES50', '/usrdata2/workspace/faster_rcnn_online/output/eleven_all_res50_scale_jitter/eleven_all_train/eleven_all_ResNet-50_faster_rcnn_anneal_stepsize450000_iter_1600000.caffemodel')}

PROTXT = {'vgg16': ('VGG16', 'vgg_demo.prototxt'), 'res50':('RES50', 'res_demo.prototxt')}


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


def calc_precision(gt_boxes, gt_clses, pred_boxes, pred_clses, thres):
  gt_boxes = np.array(gt_boxes)
  pred_boxes = np.array(pred_boxes)

  tp = 0
  fp = 0
  fn = 0

  for pred_idx, pred_box in enumerate(pred_boxes):
    max_iou = 0
    cls = None
    for gt_idx, gt_box in enumerate(gt_boxes):
      iou = boxoverlap(gt_box, pred_box)

      if max_iou < iou:
        max_iou = iou
        cls = gt_clses[gt_idx]

      #	if max_iou > float(thres) and cls == pred_clses[pred_idx]:
      if max_iou > float(thres):
        tp += 1

  fp = len(pred_boxes) - tp
  fn = len(gt_boxes) - tp

  if(fp+tp) == 0: precision = 0.
  else: precision = float(tp) / (fp + tp)

  if(tp+fn) == 0: recall = 0.
  else: recall = float(tp) / (tp + fn)

  # print 'tp : %d, fp : %d, fn : %d, precision : %f, recall : %f'%(tp,fp,fn,precision,recall)
  return precision, recall


def vis_detections(im, class_name, dets, thresh=0.5):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= thresh)[0]
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
    .format(class_name, class_name, thresh), fontsize=14)
  plt.axis('off')
  plt.tight_layout()
  plt.draw()


def detect(net, image_name):
  """Detect object classes in an image using pre-computed object proposals."""

  # Load the demo image
  im = cv2.imread(image_name)

  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(net, im)
  timer.toc()
  print ('Detection took {:.3f}s for '
       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

  # Visualize detections for each class
  CONF_THRESH = 0.8
  NMS_THRESH = 0.3
  objects_box = []
  objects_cls = []
  for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
             .astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(inds) == 0:
      continue	

  for i in inds:
    bbox = dets[i, :4]
    bbox = bbox.astype(np.int64)
    objects_cls.append(cls_ind)
    objects_box.append(bbox)
  
  return objects_cls, objects_box


def demo(net, image_name):
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
  CONF_THRESH = 0.8
  NMS_THRESH = 0.3
  for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
             .astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Faster R-CNN demo')
  parser.add_argument('--gpu', 
                      dest='gpu_id', 
                      help='GPU device id to use [0]',
                      default=0, 
                      type=int)
  parser.add_argument('--cpu', 
                      dest='cpu_mode',
                      help='Use CPU mode (overrides --gpu)',
                      action='store_true')
  parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
            choices=NETS.keys(), default='vgg16')
  parser.add_argument('--img', dest='image', help='Image to demo')
  parser.add_argument('--thres', dest='thres', help='Threshold of IoU')

  parser.add_argument('--cfg', 
                      dest='cfg_file',
                      help='optional config file',
                      default='', 
                      type=str)

  args = parser.parse_args()

  return args

if __name__ == '__main__':
  import pdb; pdb.set_trace()
  args = parse_args()
  # NOTE: get configuration from yml file or argparse
  if args.cfg_file is not None:
    # conf. from yml
    cfg_from_file(args.cfg_file)
    prototxt = cfg.TEST.PROTOTXT
    caffemodel = cfg.TEST.CAFFE_MODEL
  else:
    # conf. from argparse
    prototxt = os.path.join('/usrdata/ImageSearch/11st_DB/11st_All/prototxt', PROTXT[args.demo_net][1])
    caffemodel = NETS[args.demo_net][1]
    if args.cpu_mode:
      caffe.set_mode_cpu()
    else:
      caffe.set_device(args.gpu_id)
      caffe.set_mode_gpu()
      cfg.GPU_ID = args.gpu_id

  if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.').format(caffemodel))

  # NOTE: load faster-rcnn model
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)
  print '\n\nLoaded network {:s}'.format(caffemodel)

  # Warmup on a dummy image
  im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  for i in xrange(2):
    _, _= im_detect(net, im)

  if args.image is None :
    print args.thres
    metafile = os.path.join('/usrdata/ImageSearch/11st_DB/11st_All/Annotations', 
                            'annotations_test.txt')
    total_precision = 0
    total_recall = 0
    num_query = 0
    with open(metafile, 'r') as f:
      for line in f:
        if line is None: break
        line = line.rstrip('\n')
        word = line.split(' ')
        image_name = word[0]
        num_rois = int(word[1])
        gt_clses = []
        gt_boxes = []
        for i in range(0, num_rois):
          step = 5*i
          gt_clses.append(int(word[step+2]))
          box = [int(word[step+3]), 
                 int(word[step+4]), 
                 int(word[step+5]), 
                 int(word[step+6])]
          box = np.array(box)
          gt_boxes.append(box)

        im_path = os.path.join('/usrdata/ImageSearch/11st_DB/11st_All/Images', 
                               image_name + '.jpg')
        print im_path
        pred_clses, pred_boxes = detect(net, im_path)

      precision, recall = calc_precision(gt_boxes, 
                                         gt_clses, 
                                         pred_boxes, 
                                         pred_clses, 
                                         args.thres)
      total_precision += precision
      total_recall += recall
      num_query += 1
      print 'total_precision : %f, total_recall : %f, num_query : %d' %\
              (total_precision, total_recall, num_query)
    print 'precision : %f, recall : %f' %\
            ((float(total_precision) / num_query) * 100, 
             (float(total_recall) / num_query) * 100)
  else:
    im_path = os.path.join('/usrdata/ImageSearch/11st_DB/11st_All/Images', 
                           args.image + '.jpg')
    print im_path
    demo(net, im_path)
    plt.show()

