#!/bin/bash

echo "start compare"
# CO-DETR
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/CO-DETR/co_dino_5scale_r50_lsj_8xb2_1x_coco.py log/CO-DETR/best_coco_bbox_mAP_epoch_11.pth # --show-dir ./results/co-detr

# detr
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/detr/detr_r50_8xb2-150e_coco.py log/detr/best_coco_bbox_mAP_epoch_143.pth # --show-dir ./results/detr
# dino
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/dino/dino.py log/dino/best_coco_bbox_mAP_epoch_15.pth # --show-dir ./results/dino

# dynamic_rcnn
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py log/dynamic_rcnn/best_coco_bbox_mAP_epoch_19.pth # --show-dir ./results/dynamic_rcnn

# faster_rcnn
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py log/faster_rcnn/best_coco_bbox_mAP_epoch_4.pth # --show-dir ./results/faster_rcnn

# libra_rcnn
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py log/libra_rcnn/best_coco_bbox_mAP_epoch_6.pth # --show-dir ./results/libra_rcnn

# retinanet
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/retinanet/retinanet_r50_fpn_1x_coco.py log/retinanet/best_coco_bbox_mAP_epoch_6.pth # --show-dir ./results/retinanet

# rtmdet
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/rtmdet/rtmdet_tiny_8xb32-300e_coco.py log/rtmdet/best_coco_bbox_mAP_epoch_23.pth # --show-dir ./results/rtmdet

# vfnet
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/vfnet/vfnet_r50_fpn_1x_coco.py log/vfnet/best_coco_bbox_mAP_epoch_20.pth # --show-dir ./results/vfnet

# ViTDet
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/ViTDet/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py log/ViTDet/best_coco_bbox_mAP_epoch_31.pth # --show-dir ./results/ViTDet

echo "finish compare"