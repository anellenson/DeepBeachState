#!/bin/bash

states=("Ref" "LTT" "TBR" "RBB" "LBT")
testsites=("nbn" "duck")
imagenum=(1 5 10)
for testsite in "${testsites[@]}"
	do
	for state in "${states[@]}"
		do
		for ii in "${imagenum[@]}"
		    do
			~/pytorch/bin/python2.7 visualize_CNN.py -state $state -m resnet512_five_aug_8 -ii $ii -testsite $testsite -trainsite nbn
			wait
		done
	done
done
