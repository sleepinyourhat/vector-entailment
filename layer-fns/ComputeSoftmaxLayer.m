% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ probs, loss ] = ComputeSoftmaxLayer(in, matrix, relationRange, labels)
% Run the softmax classifier layer forward, and compute log loss if possible. 

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

unNormedProbs = exp(matrix(relationRange, :) * inPadded);
partitions = sum(unNormedProbs);
probs = bsxfun(@rdivide, unNormedProbs, partitions);

% If a correct class vector is provided, compute the objective function value.
if nargin > 3
	% Pad with ones to allow for zeros in labels, which won't contribute to cost.
	evalprobs = [ones(1, size(probs, 2)); probs];
	labels = labels + 1;
	loss = -log(evalprobs(sub2ind(size(evalprobs), labels(:), (1:length(labels))')));
else
	loss = 0;
end

end
