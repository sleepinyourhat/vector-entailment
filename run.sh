#!/bin/sh
### Script to launch an experiment on PBS.
### Work in progress.

### Job name
#PBS -N sbowman_pbs_test
### Output files
###PBS -o hello_world_job.stdout
###PBS -e hello_world_job.stderr
### Queue name 
###PBS -q dqueue
### Number of nodes 
#PBS -l nodes=4:compute#shared
# Print the default PBS server
echo PBS default server is $PBS_DEFAULT
# Print the job's working directory and enter it.
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
# Print some other environment information
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo This jobs runs on the following processors:
NODES=`cat $PBS_NODEFILE`
echo $NODES
# Compute the number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes
# Run hello_world
for NODE in $NODES; do
 ssh $NODE cat < "ABCDE" &
done
# Wait for background jobs to complete.
wait 