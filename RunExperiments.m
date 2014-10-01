% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainModel(dataflag, pretrainingFilename, expName, mbs, dim, lr, lambda, tot, sig)

echo "cd quant; mbs = [16]; name=['test-sick-only-',num2str(mbs)]; disp(name); TrainModel('sick-only', '', name, mbs, 25, 0.01, 0.001, 1);" | /afs/cs/software/bin/matlab_r2012b

echo "cd quant; mbs = [16]; name=['sick-only-',num2str(mbs)]; disp(name); TrainModel('sick-only', '', name, mbs, 25, 0.01, 0.001, 1, 0);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; mbs = [16]; name=['sick-only-2']; disp(name); TrainModel('sick-only', '', name, mbs, 25, 0.01, 0.001, 1, 0);" | /afs/cs/software/bin/matlab_r2012b

echo "cd quant; mbs = [16]; name=['sick-plus-',num2str(mbs)]; disp(name); TrainModel('sick-plus', '', name, mbs, 25, 0.01, 0.001, 1, 0);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; mbs = [16]; name=['sick-plus-full-',num2str(mbs)]; disp(name); TrainModel('sick-plus', '', name, mbs, 25, 0.01, 0.001, 1, 1);" | /afs/cs/software/bin/matlab_r2012b

echo "cd quant; mbs = [16]; name=['sick-plus-clean-',num2str(mbs)]; disp(name); TrainModel('sick-plus', '', name, mbs, 25, 0.01, 0.01, 1, 1);" | /afs/cs/software/bin/matlab_r2012b

echo "cd quant; lambda = 0.01; lr = 0.1; name=['sick-plus-clean-', num2str(lr), '-', num2str(lambda)]; disp(name); TrainModel('sick-plus', '', name, 48, 25, lr, lambda, 1, 1);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; lambda = 0.001; lr = 0.1; name=['sick-plus-clean-', num2str(lr), '-', num2str(lambda)]; disp(name); TrainModel('sick-plus', '', name, 48, 25, lr, lambda, 1, 1);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; lambda = 0.00001; lr = 0.1; name=['sick-plus-clean-', num2str(lr), '-', num2str(lambda)]; disp(name); TrainModel('sick-plus', '', name, 48, 25, lr, lambda, 1, 1);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; lambda = 0.0001; lr = 0.01; name=['sick-plus-clean-', num2str(lr), '-', num2str(lambda)]; disp(name); TrainModel('sick-plus', '', name, 48, 25, lr, lambda, 1, 1);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; lambda = 0.00001; lr = 0.01; name=['sick-plus-clean-', num2str(lr), '-', num2str(lambda)]; disp(name); TrainModel('sick-plus', '', name, 48, 25, lr, lambda, 1, 1);" | /afs/cs/software/bin/matlab_r2012b
echo "cd quant; lambda = 0.001; lr = 0.001; name=['sick-plus-clean-', num2str(lr), '-', num2str(lambda)]; disp(name); TrainModel('sick-plus', '', name, 48, 25, lr, lambda, 1, 1);" | /afs/cs/software/bin/matlab_r2012b


echo "cd quant; lambda = 0.001; lr = 0.001; name=['word-relations-', num2str(lr), '-', num2str(lambda)]; disp(name); TrainModel('word-relations', '', name, 48, 25, lr, lambda, 1, 1);" | /afs/cs/software/bin/matlab_r2012b
