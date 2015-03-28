% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function relationProbs = ComputeSoftmaxProbabilities(inVector, softmaxMatrix, relationRange)
% Run the softmax classifier layer forward.

% Note: Relation range specifies which relations are under consideration. If 
% relationRange covers the whole space of relations suported by the parameter
% matrix (i.e., relationRange = 1:size(softmaxMatrix, 1)), then this computes
% the distribution for a single normal softmax classifier. If this is not the case, then
% columns of the matrix that aren't included in relationRange are ignored, and assumed
% to not contribute to the output distribution or the partition function.

% This configuration is used to allow for one trained network to be trained using examples
% which were labeled from label sets that don't correspond exactly to the label set used
% on the test data.

if nargin < 3
	relationRange = 1:size(softmaxMatrix, 1);
end

% Add intercept term
in = [ones(1, size(inVector, 2)); inVector];

unNormedRelationProbs = exp(softmaxMatrix(relationRange, :) * in);
partitions = sum(unNormedRelationProbs);
relationProbs = bsxfun(@rdivide, unNormedRelationProbs, partitions);

end
