#/bin/bash
#PBS -l nodes=1:ppn=4 	### Request at least 4 cores
#PBS -l walltime=40:00:00	### Die after 30h
#PBS -l mem=6000MB
#PBS -W x=NACCESSPOLICY:SINGLEJOB	### Use all cores on node
#PBS -n	### Use all cores on node (alternate specification)
#PBS -m n
#PBS -q nlp

# [Don't] change to the submission directory
cd $PBS_O_WORKDIR 
echo `hostname`: $MATLABCMD 
echo `hostname` - $PBS_JOBID - $MATLABCMD >> ~/machine_assignments.txt
echo $MATLABCMD | /afs/cs/software/bin/matlab_r2012b -nodisplay
