#!/bin/bash
for((i=1;i<=3;i++));  
do      
        last=`expr $i - 1`
        #echo $last
        echo "s/0.0${last}/0.0${i}/g"
        perl -i -pe"s/0.0${last}/0.0${i}/g" /home/yan/Work/opensrc/RoutedFusion/dataset/ICL.py
 	python test_fusion.py --experiment pretrained_models/fusion/shapenet_noise_005 --test configs/tests/ICL.yaml

  	python3 our_visualize.py

	mv routed.ply finetuned2_00${i}_ICL1_0.6_512.ply
done

