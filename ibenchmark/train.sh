#! /bin/bash

DATASET_NAME=$1

python ibenchmark/train.py \
 --train_filename=data/ibenchmark/processed/${DATASET_NAME}_train_lbl.sdf \
 --test_filename=data/ibenchmark/processed/${DATASET_NAME}_test_lbl.sdf \
 --epochs=900 \
 --lr=1e-4 \
 --weight_decay=0 \
 --n_layer=6 \
 --dim=128 \
 --batch_size=32 \
 --cutoff_l=5.0 \
 --cutoff_g=5.0 \
 --seed=0 \
 --device=cuda:0 \
 --experiment_name=${DATASET_NAME}