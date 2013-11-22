#! /bin/bash
date
hostname
echo "cd /user/sbowman/quant/; TrainModel(11, 0.0007, [], {'BQ-MTI-no-some-able.tsv'}, {'BQ-MTI-all-some-able.tsv'})" | /afs/cs/software/bin/matlab_r2012b | tee log.txt


