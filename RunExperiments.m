% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
TrainModel('splitall', [], 'splitall');
TrainModel('one-mn', [], 'one-mn');
TrainModel('sub-mn', [], 'sub-mn');
TrainModel('pair-mn', [], 'pair-mn');
TrainModel('one-sn', [], 'one-sn');
TrainModel('sub-sn', [], 'sub-sn');
TrainModel('pair-sn', [], 'pair-sn');
TrainModel('one-2a', [], 'one-2a');
TrainModel('sub-2a', [], 'sub-2a');
TrainModel('pair-2a', [], 'pair-2a');

.1 .2 .4 .8
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-01ds', 0.1, 1)" | /afs/cs/software/bin/matlab_r2013b | tee 01ds.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-02ds', 0.2, 1)" | /afs/cs/software/bin/matlab_r2013b | tee 02ds.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-04ds', 0.4, 1)" | /afs/cs/software/bin/matlab_r2013b | tee 04ds.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-08ds', 0.8, 1)" | /afs/cs/software/bin/matlab_r2013b | tee 08ds.txt

echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-01d', 1, 0.1)" | /afs/cs/software/bin/matlab_r2013b | tee 01d.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-02d', 1, 0.2)" | /afs/cs/software/bin/matlab_r2013b | tee 02d.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-04d', 1, 0.4)" | /afs/cs/software/bin/matlab_r2013b | tee 04d.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-08d', 1, 0.8)" | /afs/cs/software/bin/matlab_r2013b | tee 08d.txt
