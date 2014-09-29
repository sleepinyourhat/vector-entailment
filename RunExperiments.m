% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainModel(dataflag, pretrainingFilename, expName, mbs, dim, lr, lambda, tot, sig)

echo "cd quant; mbs = [16]; name=['test-sick-only-',num2str(mbs)]; disp(name); TrainModel('sick-only', '', name, mbs, 25, 0.01, 0.001, 1);" | /afs/cs/software/bin/matlab_r2012b

echo "cd quant; mbs = [16]; name=['sick-only-',num2str(mbs)]; disp(name); TrainModel('sick-only', '', name, mbs, 25, 0.01, 0.001, 1, 0);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; mbs = [16]; name=['sick-only-2']; disp(name); TrainModel('sick-only', '', name, mbs, 25, 0.01, 0.001, 1, 0);" | /afs/cs/software/bin/matlab_r2012b

echo "cd quant; mbs = [16]; name=['sick-plus-',num2str(mbs)]; disp(name); TrainModel('sick-plus', '', name, mbs, 25, 0.01, 0.001, 1, 0);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; mbs = [16]; name=['sick-plus-full-',num2str(mbs)]; disp(name); TrainModel('sick-plus', '', name, mbs, 25, 0.01, 0.001, 1, 1);" | /afs/cs/software/bin/matlab_r2012b

