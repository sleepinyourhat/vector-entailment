% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function o = Objective(trueRelations, relationProbs, hyperParams)
% Compute the non-regularized objective for a batch.
% TODO: Add support for multiple relation sets.

o = sum(-log(relationProbs((sub2ind(size(relationProbs),trueRelations,1:size(trueRelations, 2))))));

end