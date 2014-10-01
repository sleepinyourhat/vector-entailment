vector-entailment
=================

The code for the experiments reported on in Bowman (2014) and Bowman, Potts and Manning (2014)

WARNING: I have made every effort to ensure that this code is as inefficient as possible
and violates every convention of MATLAB style. If the code could be made worse on either 
count, please contact me (sbowman@stanford.edu) and I will remedy the error.

MORE WARNING: Trees are represented as new-style objects, so you will need a fairly recent
copy of MATLAB. R2013b works.

To run a single experiment, see TrainModel.m. RunExperiments.m contains a script to run
all of the reported experiments, though many more configurations are possible if you 
edit the options and hyperparameters (no, the distinction dosen't mean much) in 
TrainModel.

The google-10000-english vocabulary is from the release compiled here:
https://github.com/first20hours/google-10000-english

Author: Samuel Bowman, sbowman@stanford.edu