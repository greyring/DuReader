#!/bin/bash
file_dir="../data/preprocessed"
train_file="--train_files $file_dir/trainset/search.train.json $file_dir/trainset/zhidao.train.json"
dev_file="--dev_files $file_dir/devset/search.dev.json $file_dir/devset/zhidao.dev.json"
test_file="--test_files $file_dir/testset/search.test1.json $file_dir/testset/zhidao.test1.json"
files="$train_file $dev_file $test_file"

python run.py --prepare $files
python run.py --train $files --algo BIDAF --epochs 4 --dropout_keep_prob 0.5 --gpu 3
python run.py --evaluate $files --algo BIDAF --gpu 3
python run.py --predict $files --algo BIDAF --gpu 3
