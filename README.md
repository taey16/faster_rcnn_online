# faster_rcnn_online
- Original post: https://github.com/rbgirshick/py-faster-rcnn

# Modifications
- Online(with threading) image-batch, roidb, regression-target generator (see. Loader.py)
- Train/Valididation

# install guide
- Follow the original post

# Caffe for Faster-RCNN
- copy roi_pooling_layer.[cpp/cu/hpp] and smooth_L1_loss_layer.[cpp/cu/hpp]

...`cd /YOUR_CAFFE_HOME/src/caffe/layers/
...wget 10.202.35.109:2596/PBrain/co_work/Moonki/roi_pooling_layer.cpp .
...wget 10.202.35.109:2596/PBrain/co_work/Moonki/roi_pooling_layer.cu .
...wget 10.202.35.109:2596/PBrain/co_work/Moonki/smooth_L1_loss_layer.cpp .
...wget 10.202.35.109:2596/PBrain/co_work/Moonki/smooth_L1_loss_layer.cu .
...cd /YOUR_CAFFE_HOME/include/caffe/layers/
...wget 10.202.35.109:2596/PBrain/co_work/Moonki/fast_rcnn_layers.hpp .`
- Modify /YOUR_CAFFE_HOME/src/caffe/proto/caffe.proto
`bla~
bla~
message LayerParameter {
  bla~
  bla~
========================= diff
  optional ROIPoolingParameter roi_pooling_param = 8266711;
  optional SmoothL1LossParameter smooth_l1_loss_param = 8266712; 
<<<<<<<<<<<<<<<<<<<<<<<<< 
}

bla~
bla~

========================= diff
// Message that stores parameters used by ROIPoolingLayer 
message ROIPoolingParameter {
  // Pad, kernel size, and stride are all given as a single value for equal
  // dimensions in height and width or as Y, X pairs.
  optional uint32 pooled_h = 1 [default = 0]; // The pooled output height
  optional uint32 pooled_w = 2 [default = 0]; // The pooled output width
  // Multiplicative spatial scale factor to translate ROI coords from their
  // input scale to the scale used when pooling
  optional float spatial_scale = 3 [default = 1]; 
} 
<<<<<<<<<<<<<<<<<<<<<<<<

bla~
bla~

======================== diff
message SmoothL1LossParameter {
  // SmoothL1Loss(x) =
  //   0.5 * (sigma * x) ** 2    -- if x < 1.0 / sigma / sigma
  //   |x| - 0.5 / sigma / sigma -- otherwise
  optional float sigma = 1 [default = 1]; 
} 
<<<<<<<<<<<<<<<<<<<<<<<<
