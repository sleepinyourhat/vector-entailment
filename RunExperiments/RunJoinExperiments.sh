% Label composition experiments

export MATLABCMD="cd quant; dataflag = 'fold1'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 1; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'fold2'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 1; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'fold3'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 1; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'fold4'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 1; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'fold5'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 1; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; dataflag = 'fold1'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 0; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'fold2'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 0; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'fold3'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 0; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'fold4'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 0; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'fold5'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 0; name='tj'; relu = 1; TrainModel(''\, 1\, @Join\, name\, dataflag\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

cd quant; dataflag = 'fold1'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 0; name='tj'; relu = 1; 
TrainModel('', 1, @Join, name, dataflag, dim, penult, td, lambda, tot, relu, dropout, 32);
