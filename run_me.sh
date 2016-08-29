#!/bin/sh

# example commend for train/val
CUDA_VISIBLE_DEVICES=0 nohup python tool/train_val_net.py --rand --cfg /works/faster_rcnn_online/cfg/faster_rcnn_end2end_train_scale_jitter.yml --output ./output > logs.log &
# prediction for mAP evaluation (non-guided)
CUDA_VISIBLE_DEVICES=0 python tool/eval.py --cfg cfg/faster_rcnn_end2end_test.yml --output ./result --predict
# prediction for mAP evaluation (guided detection)
CUDA_VISIBLE_DEVICES=0 python tool/eval.py --cfg cfg/faster_rcnn_end2end_test.yml --output ./result --guide true --predict
# mAP evaluation
CUDA_VISIBLE_DEVICES=0 python tool/eval.py --cfg cfg/faster_rcnn_end2end_test.yml --output ./result --eval
