#!/bin/bash

### Generic job script for all experiments.

#PBS -l nodes=1:ppn=6 	### Request at least 6 cores
#PBS -l walltime=199:00:00	### Die after eight days
#PBS -l mem=6000MB
#PBS -q nlp

# Usage example:
# export REMBED_FLAGS="--learning_rate 0.1"; qsub -v REMBED_FLAGS run.sh

# Change to the submission directory. (Usually ~ for me... I include a 'cd some/dir;' 
# at the start of MATLABCMD to get to my real working directory.)
cd $PBS_O_WORKDIR 

# Log what we're running and where.
echo `hostname`: $MATLABCMD 
echo `hostname` - $PBS_JOBID - $MATLABCMD >> ~/machine_assignments.txt

echo $MATLABCMD | /afs/cs/software/bin/matlab_r2014b -nodisplay
