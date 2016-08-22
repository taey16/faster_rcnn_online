# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(im, roidb, num_classes, set_id):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  # NOTE: scale jittering in training phase only
  if set_id == 'train':
    scales_list = cfg.TRAIN.SCALES
  else:
    scales_list = cfg.TEST.SCALES

  random_scale_inds = npr.randint(0, high=len(scales_list), size=num_images)
  target_size = scales_list[random_scale_inds]
    
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)
    
  # does not used in USE_RPN: True
  #rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  #fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

  # Get the input image blob, formatted for caffe
  #im_blob, im_scales = _get_image_blob(im, roidb, random_scale_inds)
  im_blob, im_scales = _get_image_blob(im, roidb, target_size)
  blobs = {'data': im_blob}

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"

  # gt boxes: (x1, y1, x2, y2, cls)
  gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
    dtype=np.float32)

  # blobs
  # data (B C H W) 
  # gt_boxes (x1, x2, y1, y2, cls_id)
  # im_info (h, w, scale)
  return blobs


#def _get_image_blob(im, roidb, scale_inds):
def _get_image_blob(im, roidb, target_size):
  """
  Builds an input blob from the images in the roidb at the specified scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in xrange(num_images):
    #target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, 
                                    cfg.PIXEL_MEANS, 
                                    target_size, 
                                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

