#!/usr/bin/env bash

DATASET=$1
PORT=${PORT:-29500}

if [ ${DATASET} == voc12 ]
then
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT --nnodes=1  run.py --dataset voc12 --train_multi_scale --gen_mask_for_multi_crop --train_multi_crop --eval --gen_seg_mask
elif [ ${DATASET} == coco14 ]
then
    python -m torch.distributed.launch --nproc-per-node=1 --master-port=$PORT --nnodes=1  run.py --dataset coco14 --train_multi_crop --eval --gen_seg_mask
else:
    echo 'False dataset'
fi
