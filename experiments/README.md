把想跑的模型从configs中复制到experiments里。
每个模型文件夹里都有多个py文件，是不同的配置，选一个用就行。
vfnet_r101_fpn_2x_coco：vfnet模型，使用resnet101作为backbone，使用fpn作为neck，2x我忘了，coco，模型再coco数据集上进行的训练。

../_base_ 改成 mmdet::_base_xxx
数据集换成wheat

更改类别数，找num_classes，改成1

为了避免当你关机，关掉终端，关掉vscode时，你正在运行的程序会停断。使用screen，从网上简单一看就行。

screen -S task 创建一个session，会话
screen -ls 查看所有会话
screen -t task 进入task会话
CTRL+a  d 退出当前会话，先同时按CTRL和a,松手，再按d
Ctrl+a  c 新建窗口，出现新 shell
CTRL+a  n 下一个窗口
CTRL+a  p 上一个窗口

进入会话后的操作，同时按CTRL+a进入命令模式，再按一个字母表示相应的操作

# 训练
```bash
CUDA_VISIBLE_DEVICES=0,1 PORT=7878 ./tools/dist_train.sh experiments/vfnet/vfnet_r50_fpn_1x_coco.py 2 --work-dir ./log/vfnet
```
这是针对同时使用多>1GPU的用法说明：
CUDA_VISIBLE_DEVICES=0,1 设定需可用哪几块GPU，从0开始编号
PORT=7878 多GPU需要相互通信，使用这个端口。如果你同时跑两个实验，每个实验用了多个GPU，端口不能一样
./tools/dist_train.sh 训练的主脚本
experiments/vfnet/vfnet_r50_fpn_1x_coco.py 你要跑的模型的配置文件
2 使用两块GPU，她会从CUDA_VISIBLE_DEVICES里选前两块
如果CUDA_VISIBLE_DEVICES=0,2,3 写2的话会使用0和2这两块GPU
--work-dir ./log/vfnet 把过程中的记录存放在哪个地方。可视化信息，日志，模型文件等

在mmdetection目录下执行这个命令

现在一定会报错，别急


# 多端项目同步，使用github
git add . & git commit -m "." & git push
git pull