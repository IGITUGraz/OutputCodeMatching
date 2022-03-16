#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

data="CIFAR10"
dir="data/"
model="resnet20_quan"
classes=10

# Train ResNet-20 models (8-bit) with output code matching on CIFAR-10
python -u main.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --ocm --code_length 16 --output_act "tanh" --outdir "results/cifar10/resnet20_quan8_OCM16/"
python -u main.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --ocm --code_length 64 --output_act "tanh" --outdir "results/cifar10/resnet20_quan8_OCM64/"

# Evaluate Stealthy T-BFA attacks on OCM defended models
python -u attack_tbfa.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --ocm --code_length 16 --output_act "tanh" --outdir "results/cifar10/resnet20_quan8_OCM16/"
python -u attack_tbfa.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --ocm --code_length 64 --output_act "tanh" --outdir "results/cifar10/resnet20_quan8_OCM64/"

# Evaluate Stealthy TA-LBF attacks on OCM defended models
python -u attack_talbf.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --ocm --code_length 16 --output_act "tanh" --outdir "results/cifar10/resnet20_quan8_OCM16/"
python -u attack_talbf.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --ocm --code_length 64 --output_act "tanh" --outdir "results/cifar10/resnet20_quan8_OCM64/"
