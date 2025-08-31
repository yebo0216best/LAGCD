#!/bin/bash
#SBATCH -o %J.out            # 脚本执行的输出将被保存在当 %J.out文件下，%j表示作业号;
#SBATCH -e %J.err            # 脚本执行的错误日志将被保存在当 %J.err文件下，%j表示作业号;
#SBATCH -p gpu_v100      # 作业提交的指定队列/分区为normal_test；
#SBATCH -J i100_la    # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1            # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1;
#SBATCH --gres=gpu:1         # 申请一块GPU卡(不需要指定-n 或者--ntasks-per-node)
#SBATCH --cpus-per-task=3
module load anaconda3
source activate py37      #正式使用请激活自己的环境
python train.py \
    --dataset_name 'imagenet_100' \
    --print_freq 600 \
    --use_ssb_splits \
    --batch_size 128 \
    --lr 0.2 \
    --eval_funcs 'v2' \
    --memax_weight 1 \
    --use_linear \
    --mid_dim 96 \
    --adapter_scale 0.1 \
    --use_da \
    --da_scale 0.2 \
    --da_freq 200 \
    --partial 12 \
    --seed 0
