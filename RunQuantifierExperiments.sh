Quantifiers(name, dim, penult, top, lambda, tot, relu, tdrop)

export MATLABCMD="cd quant; lambda = 0.0003; dim = 15; td = 1; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 25; td = 1; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0003; dim = 20; td = 2; penult = 75; dropout = 0.9; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
