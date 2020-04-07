#!/bin/bash

states=("Ref" "LTT" "TBR" "RBB" "LBT")
testsites=("nbn")
imagenum=(1 5 10)
for testsite in "${testsites[@]}"
	do
	for state in "${states[@]}"
		do
		for ii in "${imagenum[@]}"
		    do
			~/pytorch/bin/python2.7 visualize_CNN.py -state $state -m resnet512_percentage_0.05_0 -ii $ii -testsite $testsite -trainsite nbn_duck
			wait
		done
	done
done
