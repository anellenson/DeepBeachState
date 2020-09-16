#!/bin/bash

states=("Ref" "LTT" "TBR" "RBB" "LBT")
testsites=("duck" "nbn")
imagenum=(1 5 10 15 20)
modelnum=(7)
for testsite in "${testsites[@]}"
	do
	for state in "${states[@]}"
		do
		for ii in "${imagenum[@]}"
		    do
		    for mm in "${modelnum[@]}"
		    do
			~/pytorch/bin/python2.7 visualize_CNN.py -state $state -m resnet512_five_aug_trainloss_$mm -ii $ii -testsite $testsite -trainsite nbn_duck 
			wait
		    done
		done
	done
done
