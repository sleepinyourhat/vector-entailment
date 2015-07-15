export MATLABCMD="cd quant; dataflag = 'sat'; rand = 1; et = 1; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel('', 1, @SAT, name, dataflag, dim, td, lambda, comp, rand, et);" ; nlpsub -c4 -qnlp -m4gb  -ppreemptable ./quant/run.sh --extra-qsub-args="-v MATLABCMD "
export MATLABCMD="cd quant; dataflag = 'sat'; rand = 1; et = 0; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel('', 1, @SAT, name, dataflag, dim, td, lambda, comp, rand, et);" ; nlpsub -c4 -qnlp -m4gb  -ppreemptable ./quant/run.sh --extra-qsub-args="-v MATLABCMD "
export MATLABCMD="cd quant; dataflag = 'sat'; rand = 0; et = 0; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel('', 1, @SAT, name, dataflag, dim, td, lambda, comp, rand, et);" ; nlpsub -c4 -qnlp -m4gb  -ppreemptable ./quant/run.sh --extra-qsub-args="-v MATLABCMD "

export MATLABCMD="cd quant; dataflag = 'sat'; rand = 1; et = 1; lambda = 0.003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel('', 1, @SAT, name, dataflag, dim, td, lambda, comp, rand, et);" ; nlpsub -c4 -qnlp -m4gb  -ppreemptable ./quant/run.sh --extra-qsub-args="-v MATLABCMD "
export MATLABCMD="cd quant; dataflag = 'sat'; rand = 1; et = 1; lambda = 0.00003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel('', 1, @SAT, name, dataflag, dim, td, lambda, comp, rand, et);" ; nlpsub -c4 -qnlp -m4gb  -ppreemptable ./quant/run.sh --extra-qsub-args="-v MATLABCMD "

export MATLABCMD="cd quant; dataflag = 'sat'; rand = 1; et = 1; lambda = 0.001; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest-5k'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat'; rand = 1; et = 1; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel('', 1, @SAT, name, dataflag, dim, td, lambda, comp, rand, et);" ;   qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat'; rand = 1; et = 1; lambda = 0.0001; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest-5k'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; dataflag = 'sat-l'; rand = 1; et = 1; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel('', 1, @SAT, name, dataflag, dim, td, lambda, comp, rand, et);"; qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; dataflag = 'sat-ls'; rand = 1; et = 1; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest-5k'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat-ls'; rand = 1; et = 1; lambda = 0.003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest-5k'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat-ls'; rand = 1; et = 1; lambda = 0.03; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest-5k'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; dataflag = 'sat-4'; rand = 1; et = 1; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat-4'; rand = 1; et = 1; lambda = 0.003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat-4'; rand = 1; et = 1; lambda = 0.03; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh



export MATLABCMD="cd quant; dataflag = 'sat-4'; rand = 1; et = 1; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat-4'; rand = 1; et = 1; lambda = 0.000003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; dataflag = 'sat-4'; rand = 0; et = 0; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat-4'; rand = 0; et = 0; lambda = 0.000003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat-4'; rand = 0; et = -1; lambda = 0.000003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh

export MATLABCMD="cd quant; dataflag = 'sat-3-25'; rand = 0; et = 0; lambda = 0.0003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest-b'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh
export MATLABCMD="cd quant; dataflag = 'sat-3-25'; rand = 0; et = 0; lambda = 0.000003; dim = 50; td = 3; penult = 100; comp = 2; name='/scr/sbowman/sattest-b'; TrainModel(''\, 1\, @SAT\, name\, dataflag\, dim\, td\, lambda\, comp\, rand\, et);" ;  qsub -v MATLABCMD quant/run.sh


% fit perfectly at 0003
% random not hurting yet
