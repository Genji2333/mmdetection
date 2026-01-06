#!/bin/bash

echo "start compare"

# CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/ViTDet/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py 2 --work-dir ./log_crabapple/ViTDet ; 
# echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/rtmdet/rtmdet_tiny_8xb32-300e_coco.py 2 --work-dir ./log_crabapple/rtmdet ; 
echo "****************************************" ;  

# CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/retinanet/retinanet_r50_fpn_1x_coco.py 2 --work-dir ./log_crabapple/retinanet ; 
# echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/dino/dino.py 2 --work-dir ./log_crabapple/dino ; 
echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py 2 --work-dir ./log_crabapple/faster_rcnn ; 
echo "****************************************" ;  

# CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/resnest/cascade-mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco.py 2 --work-dir ./log_crabapple/resnest ; 
# echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/vfnet/vfnet_r50_fpn_1x_coco.py 2 --work-dir ./log_crabapple/vfnet ; 
echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py 2 --work-dir ./log_crabapple/dynamic_rcnn ; 
echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py 2 --work-dir ./log_crabapple/libra_rcnn ; 
echo "****************************************" ; 

# CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/detr/detr_r50_8xb2-150e_coco.py 2 --work-dir ./log_crabapple/detr ; 
# echo "****************************************" ;  

# CUDA_VISIBLE_DEVICES=2,3 PORT=7878 ./tools/dist_train.sh experiments_crabapple/CO-DETR/co_dino_5scale_r50_lsj_8xb2_1x_coco.py 2 --work-dir ./log_crabapple/CO-DETR ; 
# echo "****************************************" ;  

echo "finish compare"