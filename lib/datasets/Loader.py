
import os
import sys
import cv2
import random
import numpy as np
import cPickle
import scipy.sparse
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

"""
Online data loader class which is acting as the 
roidb.py in original py-faster-rcnn @taey16
taey1600@gmail.com
"""

class Loader:

  def __init__(self, data_path, image_set):
    self.data_path = data_path
    self.image_set = image_set
    self.image_path_prefix = \
      os.path.join(self.data_path, 'Images/%s.jpg')
    self.annotation_filename = \
      os.path.join(self.data_path, 'Annotations/annotations_' + self.image_set + '.txt')
    self.load_list_im_roi()

    # NOTE: set it is train-loader or val-loader
    # scale jitter and flip-jitter only works with train-loader
    self.set_id = image_set


  def load_list_im_roi(self):
    self.list_image_roi = [entry.strip().split(' ') \
      for entry in open(self.annotation_filename, 'r')]


  def load_roi(self, image_index):
    gt_roi = []
    elements = self.list_image_roi[image_index]
    image_path = self.image_path_prefix % elements[0]
    num_objs = int(elements[1])
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

    for i in range(0, num_objs):
      x1 = float(elements[2+i*5+1])
      y1 = float(elements[2+i*5+2])
      x2 = float(elements[2+i*5+3])
      y2 = float(elements[2+i*5+4])
      cls= int(elements[2+i*5])
      boxes[i, :] = [x1, y1, x2, y2]
      gt_classes[i] = cls
      overlaps[i, cls] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)            
    gt_roi.append({'image': image_path,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'flipped': False} )

    return gt_roi


  def flip_roi(self, orig_boxes, width):
    boxes = orig_boxes.copy()
    oldx1 = boxes[:, 0].copy()
    oldx2 = boxes[:, 2].copy()

    ex_idx = np.where(oldx2>=width)
    oldx2[ex_idx] = width-1

    boxes[:, 0] = width - oldx2 - 1
    boxes[:, 2] = width - oldx1 - 1
    assert (boxes[:, 2] >= boxes[:, 0]).all()

    return boxes


  def prepare_roidb(self, roidb, height, width):
    roidb['width'] = width
    roidb['height']= height
    # need gt_overlaps as a dense array for argmax from sparse matrix.
    gt_overlaps = roidb['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb['max_classes'] = max_classes
    roidb['max_overlaps']= max_overlaps

    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)

    return roidb


  def load_im_and_roi(self, image_index):
    filename = self.image_path_prefix % self.list_image_roi[image_index][0]
    #print(filename); sys.stdout.flush()
    im = cv2.imread(filename)
    height= im.shape[0]
    width = im.shape[1]
    roidb = self.load_roi(image_index)
    # NOTE: flip jittering in training phase ONLY
    if self.set_id == 'train':
      flipped = random.randint(0, 1)
      if flipped: 
        im = im[:,::-1,:]
        roidb[0]['boxes'] = self.flip_roi(roidb[0]['boxes'], width)
        roidb[0]['flipped'] = True

    roidb[0] = self.prepare_roidb(roidb[0], height, width)
    return im, roidb


  def get_bbox_regression_target_mean_and_std(self):
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      # Use fixed / precomputed "means" and "stds" instead of empirical values
      self.means= np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS),(self.num_classes, 1))
      self.stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self.num_classes, 1))
    else:
      # Compute values needed for means and stds
      # var(x) = E(x^2) - E(x)^2
      raise NotImplementedError

    print 'bbox target means:'
    print self.means
    print self.means[1:, :].mean(axis=0) # ignore bg class
    print 'bbox target stdevs:'
    print self.stds
    print self.stds[1:, :].mean(axis=0) # ignore bg class
    sys.stdout.flush()

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return self.means.ravel(), self.stds.ravel()


  def add_bbox_regression_targets(self, roidb):
    """Add information needed to train bounding-box regressors."""
    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
      roi = roidb[im_i]['boxes']
      max_overlaps= roidb[im_i]['max_overlaps']
      max_classes = roidb[im_i]['max_classes']
      roidb[im_i]['bbox_targets'] = self._compute_targets(roi, max_overlaps, max_classes)

    # Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
      #print "Normalizing targets"
      for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in xrange(1, num_classes):
          cls_inds = np.where(targets[:, 0] == cls)[0]
          roidb[im_i]['bbox_targets'][cls_inds, 1:] -= self.means[cls, :]
          roidb[im_i]['bbox_targets'][cls_inds, 1:] /= self.stds[cls, :]
    else:
      print "NOT normalizing targets"

    return roidb


  def _compute_targets(self, rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
      np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
      np.ascontiguousarray(rois[gt_inds, :], dtype=np.float)
    )

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)

    return targets

