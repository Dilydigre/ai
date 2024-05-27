#! /bin/bash

mkdir dataset/train
mkdir dataset/test


for img in dataset/img_align_celeba/01*.jpg; do 
	mv $img dataset/test/ 
done 

for img in dataset/img_align_celeba/02*.jpg; do 
	mv $img dataset/test/ 
done 

for img in dataset/img_align_celeba/*.jpg; do 
	mv $img dataset/train/
done 

