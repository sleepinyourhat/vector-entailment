% Quantification and negation experiments

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel('', 1, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
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

export MATLABCMD="cd quant; lambda = 0.00001; dim = 25; td = 4; penult = 75; dropout = 1; tot = 2; name='ql2'; eyes = 0.5; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, eyes\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh -q john

export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 2\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 4\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='tq'; relu = 1; TrainModel(''\, 5\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 15; td = 2; penult = 75; dropout = 1; tot = 1; name='newinit'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 0);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 15; td = 2; penult = 75; dropout = 1; tot = 1; name='newinit'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 1);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 15; td = 2; penult = 75; dropout = 1; tot = 1; name='newinit'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 100);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 15; td = 2; penult = 75; dropout = 1; tot = 1; name='newinit'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 1000);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 15; td = 2; penult = 75; dropout = 1; tot = 2; name='newinit'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 1);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 15; td = 2; penult = 75; dropout = 1; tot = 2; name='newinit'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 2);" ; qsub -v MATLABCMD quant/run.sh



export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; td = 2; penult = 75; dropout = 1; tot = 2; name='newinit'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 1);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; td = 2; penult = 75; dropout = 1; tot = 2; name='newinit'; relu = 1; TrainModel(''\, 3\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 2);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; td = 1; penult = 75; dropout = 1; tot = -1; name='latcomp'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; td = 1; penult = 75; dropout = 1; tot = 0; name='latcomp'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; td = 1; penult = 75; dropout = 1; tot = 1; name='latcomp'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; td = 1; penult = 75; dropout = 1; tot = 2; name='latcomp'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; td = 1; penult = 75; dropout = 1; tot = 3; name='latcomp'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 50; td = 1; penult = 75; dropout = 1; tot = 4; name='latcomp'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; lambda = 0.0001; dim = 20; td = 1; penult = 75; dropout = 1; tot = 0; name='latcomp-b'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, dropout);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; lambda = 0.0001; dim = 20; td = 1; penult = 75; dropout = 1; tot = 1; name='latcomp-b'; relu = 0; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, dropout);" ; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant-naacl; lambda = 0.003; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'dev'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'dev'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'dev'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.0003; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'dev'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.00003; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'dev'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.00001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'dev'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh


export MATLABCMD="cd quant-naacl; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'f1'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'f2'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'f3'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'f4'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant-naacl; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = -1; name='/scr/sbowman/vintage-quant-2'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, 'f5'\, dim\, penult\, td\, lambda\, tot\, 1\, dropout\, 32);" ; qsub -v MATLABCMD quant/run.sh

Quantifiers(name, dim, penult, top, lambda, tot, relu, tdrop, mbs)


lambda = 0.0001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 0; name='newinit'; 
TrainModel('', 1, @Quantifiers, name, dim, penult, td, lambda, tot, dropout);
