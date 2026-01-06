import matplotlib
matplotlib.rcParams['font.family'] = 'AR PL UKai CN'

# 输入父目录路径
root_dir = "/icislab/volume1/liuxiaolong/mmdetection/log"

import os
import json

# 遍历root_dir下所有A/B层级，读取scalars.json，记录A/B层级名
all_data = []
for ab_name in os.listdir(root_dir):
    ab_path = os.path.join(root_dir, ab_name)
    if not os.path.isdir(ab_path):
        continue
    # 只处理A/B层级
    for xxx in os.listdir(ab_path):
        xxx_path = os.path.join(ab_path, xxx)
        vis_data_path = os.path.join(xxx_path, 'vis_data')
        scalars_path = os.path.join(vis_data_path, 'scalars.json')
        if os.path.isfile(scalars_path):
            with open(scalars_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        d['group'] = ab_name
                        all_data.append(d)
print(f'共读取{len(all_data)}条记录，分属{len(set(d["group"] for d in all_data))}组')

import pandas as pd
from collections import defaultdict

# 按group和step合并，优先保留后出现的同step数据
group_step_dict = defaultdict(dict)
for d in all_data:
    group = d['group']
    step = d.get('step', None)
    if step is not None:
        group_step_dict[(group, step)].update(d)
df = pd.DataFrame(list(group_step_dict.values()))
df = df.sort_values(['group', 'step']).reset_index(drop=True)
df.head()

import matplotlib.pyplot as plt

# loss对比
plt.figure(figsize=(12, 5))
for group, dfg in df.groupby('group'):
    if 'loss' in dfg.columns:
        plt.plot(dfg['step'], dfg['loss'], label=f'{group} loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.title('Loss 曲线对比')
plt.legend()
plt.show()

# acc对比
plt.figure(figsize=(12, 5))
for group, dfg in df.groupby('group'):
    if 'acc' in dfg.columns:
        plt.plot(dfg['step'], dfg['acc'], label=f'{group} acc')
plt.xlabel('step')
plt.ylabel('acc')
plt.title('Accuracy 曲线对比')
plt.legend()
plt.show()

# lr对比
plt.figure(figsize=(12, 5))
for group, dfg in df.groupby('group'):
    if 'lr' in dfg.columns:
        plt.plot(dfg['step'], dfg['lr'], label=f'{group} lr')
plt.xlabel('step')
plt.ylabel('lr')
plt.title('Learning Rate 曲线对比')
plt.legend()
plt.show()

# bbox_mAP对比
if 'coco/bbox_mAP' in df.columns:
    plt.figure(figsize=(12, 5))
    for group, dfg in df.groupby('group'):
        plt.plot(dfg['step'], dfg['coco/bbox_mAP'], label=f'{group} bbox_mAP')
    plt.xlabel('step')
    plt.ylabel('bbox_mAP')
    plt.title('COCO bbox_mAP 曲线对比')
    plt.legend()
    plt.show()
    
    
# mAP相关曲线分组对比
map_keys = ['coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco/bbox_mAP_l', 'coco/bbox_mAP_m', 'coco/bbox_mAP_s']
for k in map_keys:
    if k in df.columns:
        plt.figure(figsize=(12, 6))
        for group, dfg in df.groupby('group'):
            plt.plot(dfg['step'], dfg[k], label=f'{group} {k}')
        plt.xlabel('step')
        plt.ylabel(k)
        plt.title(f'{k} 曲线对比')
        plt.legend()
        plt.show()
        
        
# loss分量分组对比
loss_keys = ['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox']
for k in loss_keys:
    if k in df.columns:
        plt.figure(figsize=(12, 6))
        for group, dfg in df.groupby('group'):
            plt.plot(dfg['step'], dfg[k], label=f'{group} {k}')
        plt.xlabel('step')
        plt.ylabel(k)
        plt.title(f'{k} 曲线对比')
        plt.legend()
        plt.show()
        
        
# 按epoch分组对比loss和mAP等指标（每个epoch最后一个step）
if 'epoch' in df.columns:
    for group, dfg in df.groupby('group'):
        df_epoch = dfg.sort_values('step').groupby('epoch').tail(1)
        plt.figure(figsize=(12, 5))
        plt.plot(df_epoch['epoch'], df_epoch['loss'], label=f'{group} loss')
        if 'coco/bbox_mAP' in df_epoch.columns:
            plt.plot(df_epoch['epoch'], df_epoch['coco/bbox_mAP'], label=f'{group} bbox_mAP')
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.title(f'{group} 每个epoch的loss和bbox_mAP')
        plt.legend()
        plt.show()
        
# 分组统计主要指标的最大/最小/平均值
metrics = ['loss', 'acc', "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75", "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l"]
for group, dfg in df.groupby('group'):
    print(f'==== {group} ====')
    for m in metrics:
        if m in dfg.columns:
            print(f'{m}: max={dfg[m].max():.4f}, min={dfg[m].min():.4f}, mean={dfg[m].mean():.4f}')
            
# 生成分组统计表格并保存到root_dir/summary.txt
import numpy as np
from tabulate import tabulate

metrics = ['loss', 'acc', "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75", "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l"]
rows = []
for group, dfg in df.groupby('group'):
    row = [group]
    for m in metrics:
        if m in dfg.columns:
            vals = dfg[m].replace([np.inf, -np.inf], np.nan)
            row.extend([f'{vals.max():.4f}', f'{vals.min():.4f}', f'{vals.mean():.4f}'])
        else:
            row.extend(['', '', ''])
    rows.append(row)
header = ['group'] + sum([[f'{m}_max', f'{m}_min', f'{m}_mean'] for m in metrics], [])
table_str = tabulate(rows, headers=header, tablefmt='grid')
print(table_str)
with open(os.path.join(root_dir, 'summary.txt'), 'w') as f:
    f.write(table_str)