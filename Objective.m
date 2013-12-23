function o = Objective(trueRelation, relationProbs)
% Compute the non-regularized objective for a single example.

o = -log(relationProbs(trueRelation));

end