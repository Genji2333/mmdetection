_base_ = 'tood_r50_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)
data_root = '/icislab/volume1/liuxiaolong/mmdetection/data/splitedpear/'
metainfo = {
    'classes': ('pear'),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}


train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/images/')))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/images/')))
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/images/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

default_hooks = dict(
    
    logger=dict(type='LoggerHook', interval=1), #  多久打印一次日志
)
    


load_from = '/icislab/volume1/liuxiaolong/mmdetection/configs/tood/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth'