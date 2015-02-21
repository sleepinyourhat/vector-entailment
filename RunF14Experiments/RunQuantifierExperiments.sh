% Quantification and negation experiments

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

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 2; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh


export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; eyes = 0.1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; eyes = 0.9; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; eyes = 0.1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='tq'; eyes = 0.9; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 3; name='tq'; eyes = 0.1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 3; name='tq'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 3; name='tq'; eyes = 0.9; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john

export MATLABCMD="cd quant; lambda = 0.01; dim = 25; td = 1; penult = 75; dropout = 1; tot = 3; name='ql'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 3; name='ql'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 3; name='ql'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john

export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 1; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 2; penult = 75; dropout = 1; tot = 1; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 1; penult = 75; dropout = 0.7; tot = 1; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john

export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 2; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 2; penult = 75; dropout = 1; tot = 2; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 1; penult = 75; dropout = 0.7; tot = 2; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john

export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 3; penult = 75; dropout = 1; tot = 1; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 3; penult = 75; dropout = 1; tot = 2; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john

export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 4; penult = 75; dropout = 1; tot = 1; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john
export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 4; penult = 75; dropout = 1; tot = 2; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john



export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 2\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 4\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 5\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
