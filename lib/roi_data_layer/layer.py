# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.
RoIDataLayer implements a Caffe Python layer.
"""

import sys
import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
import atexit


class RoIDataLayer(caffe.Layer):
  """Fast R-CNN data layer used for training."""


  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    self._perm = \
      np.random.permutation(np.arange(len(self._loader.list_image_roi)))
    self._cur = 0


  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._loader.list_image_roi):
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    self._cur += cfg.TRAIN.IMS_PER_BATCH
    return db_inds


  def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch."""
    if cfg.TRAIN.USE_PREFETCH:
      return self._blob_queue.get()
    else:
      db_inds = self._get_next_minibatch_inds()
      im, minibatch_db = self._loader.load_im_and_roi(db_inds)
      minibatch_db = self._loader.add_bbox_regression_targets(minibatch_db)
      return get_minibatch(im, minibatch_db, self._num_classes, self._loader.set_id)
      

  def set_loader(self, loader):
    """Set the data loader to be used by this layer during training."""
    self._loader = loader
    self._shuffle_roidb_inds()
    if cfg.TRAIN.USE_PREFETCH:
      self._blob_queue = Queue(10)
      self._prefetch_process = \
        BlobFetcher(self._blob_queue, self._loader, self._loader.num_classes)
      self._prefetch_process.start()
      def cleanup():
        print 'Terminating BlobFetcher'; sys.stdout.flush()
        self._prefetch_process.terminate()
        self._prefetch_process.join()
      atexit.register(cleanup)
      

  def setup(self, bottom, top):
    """Setup the RoIDataLayer."""
    # parse the layer parameter string, which must be valid YAML
    #layer_params = yaml.load(self.param_str_)
    layer_params = yaml.load(self.param_str)
    self._num_classes = layer_params['num_classes']
    self._name_to_top_map = {}

    # data blob: holds a batch of N images, each with 3 channels
    idx = 0
    top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
      max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
    self._name_to_top_map['data'] = idx
    idx += 1

    top[idx].reshape(1, 3)
    self._name_to_top_map['im_info'] = idx
    idx += 1

    top[idx].reshape(1, 4)
    self._name_to_top_map['gt_boxes'] = idx
    idx += 1

    print 'RoiDataLayer: name_to_top:', self._name_to_top_map
    assert len(top) == len(self._name_to_top_map)


  def forward(self, bottom, top):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch()

    for blob_name, blob in blobs.iteritems():
      top_ind = self._name_to_top_map[blob_name]
      # Reshape net's input blobs
      top[top_ind].reshape(*(blob.shape))
      # Copy data into net's input blobs
      top[top_ind].data[...] = blob.astype(np.float32, copy=False)


  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass


  def reshape(self, bottom, top):
    """Reshaping happens during the call to forward."""
    pass


class BlobFetcher(Process):
  """Experimental class for prefetching blobs in a separate process."""

  def __init__(self, queue, loader, num_classes):
    super(BlobFetcher, self).__init__()
    self._queue = queue
    self._loader = loader
    self._num_classes = num_classes
    self._perm = None
    self._cur = 0
    self._shuffle_roidb_inds()
    # fix the random seed for reproducibility
    np.random.seed(cfg.RNG_SEED)


  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    self._perm = np.random.permutation(np.arange(len(self._loader.list_image_roi)))
    self._cur = 0


  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._loader.list_image_roi):
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    self._cur += cfg.TRAIN.IMS_PER_BATCH
    return db_inds


  def run(self):
    print 'BlobFetcher started'; sys.stdout.flush()
    while True:
      db_inds = self._get_next_minibatch_inds()
      im, minibatch_db = self._loader.load_im_and_roi(db_inds)
      minibatch_db = self._loader.add_bbox_regression_targets(minibatch_db)
      blobs = get_minibatch(im, minibatch_db, self._num_classes)
      self._queue.put(blobs)


