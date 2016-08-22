#!/bin/sh
# example commend for train/val
CUDA_VISIBLE_DEVICES=1 nohup python tool/train_val_net.py --rand --cfg /works/faster_rcnn_online/cfg/faster_rcnn_end2end_train_scale_jitter.yml --output ./output > logs.log &
# mAP evaluation
CUDA_VISIBLE_DEVICES=0 python tool/eval.py --cfg cfg/faster_rcnn_end2end_test.yml
