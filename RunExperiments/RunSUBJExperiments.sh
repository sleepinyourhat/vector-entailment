export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; ed = 200; td = 2; penult = 75; dropout = [0.5\, 0.5]; tot = 2; collo = 1; dataflag='sst-expanded'; name='/scr/sbowman/subj-init'; TrainModel(''\, 1\, @SUBJ\, name\, dataflag\, ed\, dim\, td\, penult\, lambda\, tot\, dropout(1)\, dropout(2)\, collo\, 0\, 0\, 0\, 0.1);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.000001; dim = 50; ed = 200; td = 2; penult = 75; dropout = [0.5\, 0.5]; tot = 2; collo = 1; dataflag='sst-expanded'; name='/scr/sbowman/subj-init'; TrainModel(''\, 1\, @SUBJ\, name\, dataflag\, ed\, dim\, td\, penult\, lambda\, tot\, dropout(1)\, dropout(2)\, collo\, 0\, 0\, 0\, 0.1);" ; qsub -v MATLABCMD quant/run.sh  
export MATLABCMD="cd quant; lambda = 0.000001; dim = 20; ed = 200; td = 2; penult = 75; dropout = [0.5\, 0.5]; tot = 2; collo = 1; dataflag='sst-expanded'; name='/scr/sbowman/subj-init'; TrainModel(''\, 1\, @SUBJ\, name\, dataflag\, ed\, dim\, td\, penult\, lambda\, tot\, dropout(1)\, dropout(2)\, collo\, 0\, 0\, 0\, 0.1);" ; qsub -v MATLABCMD quant/run.sh  



																																									function [ hyperParams, options, wordMap, relationMap ] = SST(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, collo, conD, curr, mdn, ccs)

