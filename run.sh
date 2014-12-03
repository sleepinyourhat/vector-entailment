#/bin/bash
#PBS -l nodes=1:ppn=4 	### Request at least 4 cores
#PBS -l walltime=40:00:00	### Die after 30h
#PBS -l mem=6000MB
#PBS -W x=NACCESSPOLICY:SINGLEJOB	### Use all cores on node
#PBS -n	### Use all cores on node (alternate specification)
#PBS -m n
#PBS -q nlp

# Usage example:
# export MATLABCMD="cd quant; lambda = 0.0001; dim = 25; td = 1; penult = 75; dropout = 1; tot = 0; name='tq'; relu = 1; TrainModel(''\, 1\, @Quantifiers\, name\, dim\, penult\, td\, lambda\, tot\, relu\, dropout\, 32);"; qsub -v MATLABCMD quant/run.sh

# Change to the submission directory. (Usually ~ for me... I include a 'cd some/dir;' 
# at the start of MATLABCMD to get to my real working directory.)
cd $PBS_O_WORKDIR 

# Log what we're running and where.
echo `hostname`: $MATLABCMD 
echo `hostname` - $PBS_JOBID - $MATLABCMD >> ~/machine_assignments.txt

echo $MATLABCMD | /afs/cs/software/bin/matlab_r2012b -nodisplay
