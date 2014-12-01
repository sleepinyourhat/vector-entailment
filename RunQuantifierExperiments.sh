Quantifiers(name, dim, penult, top, lambda, tot, relu, tdrop)

export MATLABCMD="cd quant; lambda = 0.0003; dim = 15; td = 1; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 25; td = 1; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 2; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0003; dim = 15; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 4);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 1);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 30; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.003; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.00003; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 20; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 30; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 2\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 4\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 5\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 2\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 4\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 5\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 5\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 5\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

424 
445
435

john10.stanford.edu - 768424.scail.stanford.edu - cd quant; lambda = 0.0003; dim = 50; ed = 200; td = 1; penult = 75; lr = 0.02; dropout = [0.65, 0.85]; rtemult = 0; nlimult = 0; mult = 9; coll
o = 1; tscale = 1; wscale = 0.01; mbs = 32; dataflag='sick-plus-100k-ea'; name='ad'; TrainModel('', 1, @Sick, name, dataflag, ed, dim, td, penult, lambda, 1, 0, mbs, lr, dropout(1), dropout(2),
 mult, rtemult, nlimult, collo, tscale, wscale, 1, 1e-7);

 jude22.stanford.edu - 768445.scail.stanford.edu - cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel('', 5, @Quantifiers, name, dim, penult, td, lambda, tot, relu, dropout, 32);


jude28.stanford.edu - 768435.scail.stanford.edu - cd quant; dataflag = 'fold5'; lambda = 0.001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tj'; relu = 1; TrainModel('', 1, @Join
, name, dataflag, dim, penult, td, lambda, tot, relu, dropout, 32);

quant/tq-d15-pen75-top1-tot1-relu1-l0.0003-dropout1-mb32/stat_log

