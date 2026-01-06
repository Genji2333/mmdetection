#!/bin/bash


# dino
CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log_crabapple/dino/dino.py log_crabapple/dino/best_coco_bbox_mAP_epoch_26.pth --out dino.pkl

# # dynamic_rcnn
# CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log_crabapple/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py log_crabapple/dynamic_rcnn/best_coco_bbox_mAP_epoch_22.pth --out dynamic_rcnn.pkl

# # faster_rcnn
# CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log_crabapple/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py log_crabapple/faster_rcnn/best_coco_bbox_mAP_epoch_11.pth --out faster_rcnn.pkl

# # libra_rcnn
# CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log_crabapple/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py log_crabapple/libra_rcnn/best_coco_bbox_mAP_epoch_21.pth --out libra_rcnn.pkl

# # rtmdet
# CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log_crabapple/rtmdet/rtmdet_tiny_8xb32-300e_coco.py log_crabapple/rtmdet/best_coco_bbox_mAP_epoch_29.pth --out rtmdet.pkl

# # vfnet
# CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log_crabapple/vfnet/vfnet_r50_fpn_1x_coco.py log_crabapple/vfnet/best_coco_bbox_mAP_epoch_32.pth --out vfnet.pkl

# # ViTDet
# CUDA_VISIBLE_DEVICES=1,2 python tools/test.py log/ViTDet/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py log/ViTDet/best_coco_bbox_mAP_epoch_31.pth --show-dir ./results/ViTDetg

echo "finish compare"