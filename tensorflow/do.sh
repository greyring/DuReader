#!/bin/bash

#every time run this bash have to change save_dir

source ../../env/bin/activate
file_dir="../data/preprocessed"
train_file="--train_files $file_dir/trainset/search.train2.json"
dev_file="--dev_files $file_dir/devset/search.dev2.json $file_dir/devset/zhidao.dev2.json"
test1_file="$file_dir/testset/search.test1.json $file_dir/testset/zhidao.test1.json"
test2_file="$file_dir/testset/search.test2.json $file_dir/testset/zhidao.test2.json" 
test_file="--test_files $test1_file $test2_file"

test1="--test_files $test1_file --result_dir ../data/results/$modle_id/test1"
test2="--test_files $test2_file --result_dir ../data/results/$modle_id/test2"

files="$train_file $dev_file $test_file"

modle_id="4_25"
save_dir="--model_dir ../data/models/$modle_id --log_path ../data/logs/$modle_id.log"

param="--algo BIDAF --epochs 4 --batch_size 4 --embed_size 300 --gpu 0"

python run.py --prepare $files $save_dir $param
python run.py --train $files $save_dir $param
python run.py --predict $train_file $dev_file $save_dir $test1 $param
python run.py --predict $train_file $dev_file $save_dir $test2 $param
