#! /bin/bash
date
hostname
echo "cd /user/sbowman/quant/; TrainModel(11, 0.0005, {}, {}, {'BQ-MTI-all-some.tsv', 'BQ-MTI-no-all.tsv',		'MTI-MTI-Thai-animal.tsv', 'BQ-MTI-no-some.tsv',		'MTI-Thai-null.tsv', 'BQ-all-no.tsv',			'MTI-null-Thai.tsv', 'BQ-all-some.tsv',			'NBQ-most-two.tsv', 'BQ-some-no.tsv',			'NBQ-three-most.tsv', 'MQ-all-most.tsv',			'NBQ-three-two.tsv', 'MQ-all-three.tsv',         'NEG-MTI-not-Thai-null.tsv', 'MQ-all-two.tsv',			'NEG-notnot-null.tsv', 'MT-animal.tsv'})" | /afs/cs/software/bin/matlab_r2012b | tee log.txt


