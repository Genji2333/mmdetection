_base_ = [
    'mmdet::_base_/datasets/wheat.py',
    'mmdet::_base_/schedules/schedule_1x.py', 
    'mmdet::_base_/default_runtime.py',
    'mmdet::mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))

# model = dict(
#     type='MaskRCNN',
#     backbone=dict(
#         depth=18,
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
#     neck=dict(in_channels=[64, 128, 256, 512]))



# optimizer # 这个是更新模型权重的优化器设置。这些配置都是继承自文件开头的模型，这里对继承的内容进行重写。
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=None)

# learning rate # 学习率设置
max_epochs = 12
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# 比如模型跑多少轮，几轮进行一下验证，也在schedule里

train_cfg = dict(max_epochs=max_epochs) # 这里重写训练轮数

# 多久打印一次日志，一起其他不属于数据、模型与更新器的设置。模型集成自defalut_runtime