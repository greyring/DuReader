#!/bin/bash

source ../../env/bin/activate

file_dir="../data/preprocessed"
train_files="$file_dir/trainset/search.train2.json $file_dir/trainset/zhidao.train2.json"
dev_files="$file_dir/devset/search.dev2.json $file_dir/devset/zhidao.dev2.json"
test1_files="$file_dir/testset/search.test1.json $file_dir/testset/zhidao.test1.json"
test2_files="$file_dir/testset/search.test2.json $file_dir/testset/zhidao.test2.json"
files="--train_files $train_files --dev_files $dev_files --test_files $test1_files $test2_files"

model_id="4_29"
save_dir="--save_dir ../data/pre$model_id"
mkdir ../data/pre$model_id

param="--gpu 0 --epoches 20 --p_num 10 --p_len 200 --q_len 20"

python run.py --prepare $files $save_dir $param
python run.py --train $files $save_dir $param
python run.py --predict $files $save_dir $param

