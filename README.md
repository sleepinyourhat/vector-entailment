vector-entailment
=================

This folder contains code for the experiments reported on in Bowman, Potts and Manning (2014, 2015).

WARNING: I have made every effort to ensure that this code is as inefficient as possible
and violates every convention of MATLAB style. If the code could be made worse on either 
count, please contact me (sbowman@stanford.edu) and I will remedy the error.

MORE WARNING: Trees are represented as new-style objects, so you will need a fairly recent
copy of MATLAB. R2012b works.

To get started, have a look at the job launch commands in RunW15Experiments and the
config files in config/. A typical job will create a folder with two log files and 
intermittent model checkpoint files from which experiments can restart if they're killed.

Note: There aren't compiled scripts for the SynsetRelations experiment from B,P,M('14), and
a bit of the code used there is deprecated in this release. If you're interested in rerunning
that experiment and need help, contact us.

The SICK data is from the SemEval 2014 SICK challenge:
http://alt.qcri.org/semeval2014/task1/

The collapsed version of SICK was created with code from Neha Nayak, not released here.

minFunc is from Mark Schmidt, here:
http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html

The Denotation Graph data (used in this repo, but not distributed with it) is from here:
http://shannon.cs.illinois.edu/DenotationGraph/

sick_data/wcmac_data.txt was collected as part of Bill MacCartney's 2009 Stanford dissertation.

Author: Samuel Bowman, sbowman@stanford.edu