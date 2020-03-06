#!/bin/bash

states=("Ref" "LTT-B" "TBR-CD" "RBB-E" "LBT-FG")
for state in "${states[@]}"
    do
	python visualize_CNN.py -state $state -m inception_resnet_aug_fulltrained  -ii 0 -testsite nbn -trainsite nbn_duck
	wait
done
