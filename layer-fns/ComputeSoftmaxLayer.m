% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ probs, loss ] = ComputeSoftmaxLayer(in, matrix, hyperParams, labels)
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

% labels should be a B x 1 matrix of labels if only one class set is used, or a B x 2 matrix
% with the second column containing class set indices if multiple are used.

% TODO: Support unlabeled data with multiple class sets.

B = size(in, 2);
inPadded = [ones(1, B); in];

if (nargin > 3) && (size(labels, 2) == 2)
	% Multiple class set case.
	% TODO: Vectorize and/or shuttle easy cases to other version.

	loss = zeros(B, 1);
	probs = zeros(size(matrix, 1), B); % This will be padded with zeros at the end if a shorter class set is used.

	for b = 1:B
		relationRange = hyperParams.relationRanges{labels(b, 2)};
		unNormedProbs = exp(matrix(relationRange, :) * inPadded(:, b));
		partition = sum(unNormedProbs);
		probs(1:length(relationRange), b) = unNormedProbs / partition;

		% Pad with ones to allow for zeros in labels, which won't contribute to cost.
		if labels(b, 1) > 0
			loss(b) = -log(probs(labels(b, 1)));
		end
	end
else
	% Single class set case.

	unNormedProbs = exp(matrix * inPadded);
	partitions = sum(unNormedProbs);
	probs = bsxfun(@rdivide, unNormedProbs, partitions);
end

% If a correct class vector is provided, compute the objective function value.
if nargin > 3
	% Pad with ones to allow for zeros in labels, which won't contribute to cost.
	evalprobs = [ones(1, size(probs, 2)); probs];
	labels = labels + 1;
	loss = -log(evalprobs(sub2ind(size(evalprobs), labels(:, 1), (1:size(labels, 1))')));
else
	loss = 0;
end

end
