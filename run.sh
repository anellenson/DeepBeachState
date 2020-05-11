#!/bin/bash

states=("Ref" "LTT" "TBR" "RBB" "LBT")
testsites=("nbn" "duck")
imagenum=(1 5 10 15 20)
modelnum=(1)
for testsite in "${testsites[@]}"
	do
	for state in "${states[@]}"
		do
		for ii in "${imagenum[@]}"
		    do
		    for mm in "${modelnum[@]}"
		    do
			~/pytorch/bin/python2.7 visualize_CNN.py -state $state -m resnet512_five_aug_$mm -ii $ii -testsite $testsite -trainsite nbn
			wait
		    done
		done
	done
done
