#!bin/usr/bash

dataset_path='../../dataset/'
data_path='../data/'

if [ ! -e $data_path ] ; then
    mkdir $data_path
fi

gcc -o make_dataset make_dataset.c
./make_dataset $dataset_path $data_path
