_base_ = './configs/rtmdet_l_8xb32-300e_coco.py'

checkpoint = '/icislab/volume1/liuxiaolong/mmdetection/weights/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
           type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False))


#dict(
#           type='Pretrained', prefix='backbone.', checkpoint=checkpoint))