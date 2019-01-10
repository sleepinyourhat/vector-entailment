vector-entailment
=================

This is code for the experiments reported on in these papers:

https://arxiv.org/abs/1506.04834
http://aclweb.org/anthology/W/W15/W15-4002.pdf
http://www.aaai.org/ocs/index.php/SSS/SSS15/paper/download/10221/10027
https://arxiv.org/abs/1312.6192

WARNING: I have made every effort to ensure that this code is as inefficient as possible
and violates every convention of MATLAB style. If the code could be made worse on either 
count, please contact me (sbowman@stanford.edu) and I will remedy the error.

MORE WARNING: Trees are represented as new-style objects, so you will need a fairly recent
copy of MATLAB. R2012b works.

To get started, have a look at the job launch commands in RunExperiments (ideally from a release, 
since this changes all the time) and the config files in config/.

If you don't want to run jobs using PBS (or can't), you can replace the escaped commas (\,)
with plain commas in the commands in RunF14Experiments and pipe the commands into MATLAB, as here:

  echo "cd quant; dataflag = 'fold5'; lambda = 0.001; dim = 15; td = 1; penult = 75; dropout = 1; tot = 0; name='tj'; relu = 1; TrainModel('', 1, @Join, name, dataflag, dim, penult, td, lambda, tot, relu, dropout, 32);" | matlab

The SICK data is from the SemEval 2014 SICK challenge:
http://alt.qcri.org/semeval2014/task1/

minFunc is from Mark Schmidt, here:
http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html

The Denotation Graph data (used in this repo, but not distributed with it) is from here:
http://shannon.cs.illinois.edu/DenotationGraph/

sick-data/wcmac_data.txt was collected as part of Bill MacCartney's 2009 Stanford dissertation.

sst-data/ contais data from the Stanford Sentiment Treebank. More information can be found in the
README.txt file there.

Maintainer and lead author: Samuel Bowman, sbowman@stanford.edu
