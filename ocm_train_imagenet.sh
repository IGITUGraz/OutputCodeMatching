#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt

data="ImageNet"
dir="data/"
model="resnet50_quan"
classes=1000

# Train vanilla ResNet-50 models (8-bit) on ImageNet and perform finetuning to obtain OCM models
python -u main.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 -e 100 -b 256 --outdir "results/imagenet/resnet50_quan8/"
python -u main.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --ocm --code_length 1024 --output_act "tanh" --finetune --ft_path "results/imagenet/resnet50_quan8/" -e 60 -b 256 --opt "adam" --lr 1e-3 -wd 0.0 --outdir "results/imagenet/resnet50_quan8_OCM1024/"

# Evaluate Stealthy T-BFA attacks on vanilla ResNet-50 and then OCM defended models
python -u attack_tbfa.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --outdir "results/imagenet/resnet50_quan8/"
python -u attack_tbfa.py --data_dir $dir --dataset $data -c $classes --arch $model --bits 8 --ocm --code_length 1024 --output_act "tanh" --outdir "results/imagenet/resnet50_quan8_OCM1024/"
