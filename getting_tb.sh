#!/bin/bash

# Arguments
model=$1
dataset=$2

user=janselh@prontodtn.las.iastate.edu

location=runs
data=/work/LAS/jannesar-lab/SS_Clustering/SimCLR-2/$location/$model

function calling {
	echo "Download tensorboard objects from" $model/$dataset
	scp -r $user:$data/$dataset/events.* $location/$model/$dataset
	echo "Download tensorboard objects from" $model/$dataset/eval
	scp -r $user:$data/$dataset/eval/events.* $location/$model/$dataset/eval
}


if ! [ -z "$model" ]; then
	if [ -d $location/$model ]; then
		echo $location/$model "already created"
	else
		mkdir $location/$model
	fi

	echo "Download tensorboard objects from" $model
	scp -r $user:$data/events.* $location/$model



	if ! [ -z "$dataset" ]; then

		if [ -d $location/$model/$dataset ]; then
			echo $location/$model/dataset "already created"
			calling

		else
			mkdir $location/$model/$dataset
			mkdir $location/$model/$dataset/eval
			echo $location/$model/$dataset "created"
			calling

		fi

	fi
else
	scp -r $user:$data/ .
fi






# Predefined runs
#model=simclr_run1_copy
#model=attn_run13
#model=attn_run11
#model=attn_run18
#model=attn_run19
#model=attn_run25
#model=attn_run26
#model=simclr_tf2_test

#dataset=imagenet2012
#dataset=cifar10_original_eval
#dataset=oxford_iiit_pet_our_eval
#dataset=oxford_iiit_pet
