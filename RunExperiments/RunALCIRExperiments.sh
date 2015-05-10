export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.01; dim = 25; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 3; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 1; dataflag='long'; name='/scr/sbowman/alcir'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  

export MATLABCMD="cd quant; lambda = 0.003; dim = 25; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.0003; dim = 25; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 10; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 50; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  

export MATLABCMD="cd quant; lambda = 0.003; dim = 25; td = 2; penult = 75; comp = 1; dataflag='long'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 1; dataflag='long'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.0003; dim = 25; td = 2; penult = 75; comp = 1; dataflag='long'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  

export MATLABCMD="cd quant; lambda = 0.003; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.0003; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 0.25);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 1);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 3);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 8);" ; qsub -v MATLABCMD quant/run.sh  

% To run
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 1; dataflag='long'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 0);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 0);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 15; td = 2; penult = 75; comp = 1; dataflag='long'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 0);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 15; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 0);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 50; td = 2; penult = 75; comp = 1; dataflag='long'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 0);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 50; td = 2; penult = 75; comp = 1; dataflag='cv'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 0);" ; qsub -v MATLABCMD quant/run.sh  

export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 10);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 3);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 1);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, .33);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, -1);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, 0);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, -2);" ; qsub -v MATLABCMD quant/run.sh  

export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, .33);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, .11);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.001; dim = 25; td = 2; penult = 75; comp = 4; dataflag='parsetest'; name='/scr/sbowman/alcir-b'; TrainModel(''\, 1\, @ALCIR\, name\, dataflag\, dim\, td\, penult\, lambda\, comp\, .03);" ; qsub -v MATLABCMD quant/run.sh  


% b 001 0.981
% a 001 0.988
% long also best at 001
% higher dim for long?
