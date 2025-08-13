_base_ = [
    'mmdet::_base_/datasets/wheat.py',
    'mmdet::_base_/schedules/schedule_1x.py', 
    'mmdet::_base_/default_runtime.py',
    './configs/retinanet_r50_fpn.py',
    './configs/retinanet_tta.py'
]

# optimizer
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
