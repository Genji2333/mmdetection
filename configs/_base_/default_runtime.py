default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'), # 基于epoch还是iteration
    logger=dict(type='LoggerHook', interval=50), #  多久打印一次日志
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1), #  多久保存一次模型权重文件，这里是训练一轮就保存一个pth模型文件。
    sampler_seed=dict(type='DistSamplerSeedHook'), # 随机种子
    visualization=dict(type='DetVisualizationHook')) # 可视化

env_cfg = dict( # 环境变量
    cudnn_benchmark=False, 
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
