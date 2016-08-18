#!/bin/sh
CUDA_VISIBLE_DEVICES=1 nohup python tool/train_val_net.py --rand --cfg /works/faster_rcnn_online/cfg/11st_All/faster_rcnn_end2end_train_scale_jitter.yml --output ./output > logs.log &
