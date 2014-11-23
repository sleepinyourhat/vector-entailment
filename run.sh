#/bin/bash
#PBS -l nodes=1:ppn=4 	### Request at least 4 cores
#PBS -l walltime=12:00:00	### Die after 12h
#PBS -l mem=6000MB
#PBS -W x=NACCESSPOLICY:SINGLEJOB	### Use all cores on node
#PBS -n	### Use all cores on node (alternate specification)
#PBS -m n
#PBS -q nlp

# [Don't] change to the submission directory
cd $PBS_O_WORKDIR 
echo $MATLABCMD 
echo $MATLABCMD | /afs/cs/software/bin/matlab_r2012b -nodisplay
