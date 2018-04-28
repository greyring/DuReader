#!/bin/bash

#every time run this bash have to change save_dir

source ../../env/bin/activate

file_dir="../data/preprocessed"

train_file="--train_files $file_dir/trainset/search.train2.json"
dev_file="--dev_files $file_dir/devset/search.dev2.json $file_dir/devset/zhidao.dev2.json"

test1_file="$file_dir/testset/search.test1.json $file_dir/testset/zhidao.test1.json"
test2_file="$file_dir/testset/search.test2.json $file_dir/testset/zhidao.test2.json" 
test_file="--test_files $test1_file $test2_file"

files="$train_file $dev_file $test_file"

model_id="4_27"
test1="--test_files $test1_file --result_dir ../data/results/$model_id/test1"
test2="--test_files $test2_file --result_dir ../data/results/$model_id/test2"
mkdir ../data/results/$model_id
mkdir ../data/results/$model_id/test1
mkdir ../data/results/$model_id/test2

save_dir="--model_dir ../data/models/$model_id --log_path ../data/logs/$model_id.log"

param="--algo BIDAF --epochs 10 --batch_size 32 --embed_size 300 --gpu 0 --use_devset"

python run.py --prepare $dev_file $save_dir $param
#python run.py --prepare $files $save_dir $param
#python run.py --train $files $save_dir $param
#python run.py --predict $train_file $dev_file $save_dir $test1 $param
#python run.py --predict $train_file $dev_file $save_dir $test2 $param

#python run.py --resume $files $save_dir $param
