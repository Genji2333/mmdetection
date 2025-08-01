_base_ = [
    'mmdet::_base_/datasets/wheat.py',
    'mmdet::_base_/schedules/schedule_1x.py', 
    'mmdet::_base_/default_runtime.py'
]
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
    ],
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(
                _delete_=True,
                type='BalancedL1Loss',
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='CombinedSampler',
                num=512,
                pos_fraction=0.25,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)))))

# # MMEngine support the following two ways, users can choose
# # according to convenience
# # _base_.train_dataloader.dataset.proposal_file = 'libra_proposals/rpn_r50_fpn_1x_train2017.pkl'  # noqa
# train_dataloader = dict(
#     dataset=dict(proposal_file='libra_proposals/rpn_r50_fpn_1x_train2017.pkl'))

# # _base_.val_dataloader.dataset.proposal_file = 'libra_proposals/rpn_r50_fpn_1x_val2017.pkl'  # noqa
# # test_dataloader = _base_.val_dataloader
# val_dataloader = dict(
#     dataset=dict(proposal_file='libra_proposals/rpn_r50_fpn_1x_val2017.pkl'))
# test_dataloader = val_dataloader


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