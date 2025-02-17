#! /bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

for dataset_name in amide_class amide N_minus_O N_O N pharmacophore; do
    sbatch --output=outs/${dataset_name}-%j.out ibenchmark/train.sh $dataset_name
done