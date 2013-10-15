#! /bin/bash
date
hostname
echo "cd /user/sbowman/quant/; TrainModel('not-not', 300, 300)" | /afs/cs/software/bin/matlab_r2012b \
 | tee not-not-higherdim-300.txt
