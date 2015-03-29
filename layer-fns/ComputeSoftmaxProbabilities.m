% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ relationProbs, loss ] = ComputeSoftmaxProbabilities(in, matrix, relationRange, trueRelation)
% Run the softmax classifier layer forward, and compute log loss if possible. 

% TODO: Make batch compatible and merge with ComputeBatchSoftmaxProbabilities.

% Note: Relation range specifies which relations are under consideration. If 
% relationRange covers the whole space of relations suported by the parameter
% matrix (i.e., relationRange = 1:size(matrix, 1)), then this computes
% the distribution for a single normal softmax classifier. If this is not the case, then
% columns of the matrix that aren't included in relationRange are ignored, and assumed
% to not contribute to the output distribution or the partition function.

% This configuration is used to allow for one trained network to be trained using examples
% which were labeled from label sets that don't correspond exactly to the label set used
% on the test data.

if nargin < 3
	relationRange = 1:size(matrix, 1);
end

% Add intercept term
inPadded = [ones(1, size(in, 2)); in];

unNormedRelationProbs = exp(matrix(relationRange, :) * inPadded);
partitions = sum(unNormedRelationProbs);
relationProbs = bsxfun(@rdivide, unNormedRelationProbs, partitions);

% If a correct class is provided, compute the log loss.
if nargin > 3
	assert(sum(trueRelation > 0) == 1)
	for relationIndex = 1:length(trueRelation)
		if trueRelation(relationIndex) ~= 0
			loss = -log(relationProbs(trueRelation(relationIndex)));
		end
	end
else
	loss = 0;
end

end
