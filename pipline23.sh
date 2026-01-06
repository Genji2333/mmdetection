#!/bin/bash

echo "start compare"

# CUDA_VISIBLE_DEVICES=4,5 PORT=7878 ./tools/dist_train.sh experiments/ViTDet/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py 2 --work-dir ./log/ViTDet ; 
# echo "****************************************" ;  

# CUDA_VISIBLE_DEVICES=4,5 PORT=7878 ./tools/dist_train.sh experiments/rtmdet/rtmdet_tiny_8xb32-300e_coco.py 2 --work-dir ./log/rtmdet ; 
# echo "****************************************" ;  

# CUDA_VISIBLE_DEVICES=4,5 PORT=7878 ./tools/dist_train.sh experiments/retinanet/retinanet_r50_fpn_1x_coco.py 2 --work-dir ./log/retinanet ; 
# echo "****************************************" ;  

# CUDA_VISIBLE_DEVICES=4,5 PORT=7878 ./tools/dist_train.sh experiments/dino/dino.py 2 --work-dir ./log/dino ; 
# echo "****************************************" ;  

# CUDA_VISIBLE_DEVICES=4,5 PORT=7878 ./tools/dist_train.sh experiments/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py 2 --work-dir ./log/faster_rcnn ; 
# echo "****************************************" ;  

# # CUDA_VISIBLE_DEVICES=4,5 PORT=7878 ./tools/dist_train.sh experiments/resnest/cascade-mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco.py 2 --work-dir ./log/resnest ; 
# # echo "****************************************" ;  

# CUDA_VISIBLE_DEVICES=4,5 PORT=7878 ./tools/dist_train.sh experiments/vfnet/vfnet_r50_fpn_1x_coco.py 2 --work-dir ./log/vfnet ; 
# echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=8787 ./tools/dist_train.sh experiments/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py 2 --work-dir ./log/dynamic_rcnn ; 
echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=8787 ./tools/dist_train.sh experiments/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py 2 --work-dir ./log/libra_rcnn ; 
echo "****************************************" ; 

CUDA_VISIBLE_DEVICES=2,3 PORT=8787 ./tools/dist_train.sh experiments/detr/detr_r50_8xb2-150e_coco.py 2 --work-dir ./log/detr ; 
echo "****************************************" ;  

CUDA_VISIBLE_DEVICES=2,3 PORT=8787 ./tools/dist_train.sh experiments/CO-DETR/co_dino_5scale_r50_lsj_8xb2_1x_coco.py 2 --work-dir ./log/CO-DETR ; 
echo "****************************************" ;  

echo "finish compare"