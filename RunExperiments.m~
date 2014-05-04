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

echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-01ds', 0.1, 1)" | /afs/cs/software/bin/matlab_r2013b | tee 01ds.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-02ds', 0.2, 1)" | /afs/cs/software/bin/matlab_r2013b | tee 02ds.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-04ds', 0.4, 1)" | /afs/cs/software/bin/matlab_r2013b | tee 04ds.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-08ds', 0.8, 1)" | /afs/cs/software/bin/matlab_r2013b | tee 08ds.txt

echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-30d', 30, 0)" | /afs/cs/software/bin/matlab_r2013b | tee rnn30d.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-45d', 45, 0)" | /afs/cs/software/bin/matlab_r2013b | tee rnn45d.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-60d', 60, 0)" | /afs/cs/software/bin/matlab_r2013b | tee rnn60d.txt
echo "cd quant;TrainModel('pair-mn', [], 'pair-mn-75d', 75, 0)" | /afs/cs/software/bin/matlab_r2013b | tee rnn75d.txt

function TrainModel(dataflag, pretrainingFilename, expName, dim, tot)
