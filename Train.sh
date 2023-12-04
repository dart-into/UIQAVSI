#!bin/bash
for i in {1..50}
do
	  echo $i
	    filename="./savemodel/UIQAVSImodel_${i}.pth"
	      python train.py --save_path $filename --current_epoch $i
 done

