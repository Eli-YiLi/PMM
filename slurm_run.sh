#!/usr/bin/env bash

DATASET=$1
PARTITION=$2

if [ ${DATASET} == voc12 ]
then
    srun -p ${PARTITION} --job-name=python --gres=gpu:8 --ntasks=1 --ntasks-per-node=1 python3 run.py --dataset voc12 --train_multi_scale --gen_mask_for_multi_crop --train_multi_crop --eval --gen_seg_mask
elif [ ${DATASET} == coco14 ]
then
    srun -p ${PARTITION} --job-name=python --gres=gpu:8 --ntasks=1 --ntasks-per-node=1 python3 run.py --dataset coco14 --train_multi_crop --eval --gen_seg_mask
else:
    echo 'false dataset'
fi

