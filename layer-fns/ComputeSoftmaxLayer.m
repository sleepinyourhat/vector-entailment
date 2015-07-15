% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ probs, loss, probCorrect ] = ComputeSoftmaxLayer(in, matrix, hyperParams, labels, multipliers, active, score_dist)
% Run the softmax classifier layer forward, and compute log loss if possible. 

% Note: Label range specifies which labels are under consideration. If 
% labelRange covers the whole space of labels suported by the parameter
% matrix (i.e., labelRange = 1:size(matrix, 1)), then this computes
% the distribution for a single normal softmax classifier. If this is not the case, then
% columns of the matrix that aren't included in labelRange are ignored, and assumed
% to not contribute to the output distribution or the partition function.

% This configuration is used to allow for one trained network to be trained using examples
% which were labeled from label sets that don't correspond exactly to the label set used
% on the test data.

% labels should be a B x 1 matrix of labels if only one class set is used, or a B x 2 matrix
% with the second column containing class set indices if multiple are used.

% dish should be a C x B matrix reflecting a set of distributions over C classes.

% If the matrix is empty, we assume that the input vector already contains the appropriate features.

% active should be a boolean matrix of the same size as in, idicating wheather to use that node
% in computing the denominator. 

% TODO: Support unlabeled data with multiple class sets.

B = size(in, 2);

if (nargin > 3) && ~isempty(labels) && (size(labels, 2) == 2)
	% Multiple class set case.
	% TODO: Vectorize and/or shuttle easy cases to other version.

	if ~isempty(matrix)
		inPadded = [ones([1, size(in, 2)], 'like', in); in];
		D = size(matrix, 1);
	else
		D = size(in, 1);
	end

	loss = zeros(B, 1);
	probs = zeros([D, B], 'like', in); % This will be padded with zeros at the end if a shorter class set is used.

	for b = 1:B
		labelRange = hyperParams.labelRanges{labels(b, 2)};
		if ~isempty(matrix)
			unNormedProbs = exp(matrix(labelRange, :) * inPadded(:, b));
		else
			unNormedProbs = exp(in);
		end

		if nargin > 5 && ~isempty(active)
			unNormedProbs = unNormedProbs .* active;
		end

		partition = sum(unNormedProbs);
		probs(1:length(labelRange), b) = unNormedProbs / partition;

		% Pad with ones to allow for zeros in labels, which won't contribute to cost.
		if labels(b, 1) > 0
			loss(b) = gather(-log(probs(labels(b, 1))));
		end
	end
else
	% Single class set case.
	if ~isempty(matrix)
		inPadded = [ones([1, size(in, 2)], 'like', in); in];
		unNormedProbs = exp(matrix * inPadded);
	else
		unNormedProbs = exp(in);
	end

	if nargin > 5 && ~isempty(active)
		unNormedProbs = unNormedProbs .* active;
	end

	partitions = sum(unNormedProbs);
	probs = bsxfun(@rdivide, unNormedProbs, partitions);
end

% If a correct class vector is provided, compute the objective function value.
if nargin > 3 && ~isempty(labels)
	% Pad with ones to allow for zeros in labels, which won't contribute to cost.
	evalprobs = [ones([1, size(probs, 2)], 'like', probs); probs];
	labels = labels + 1;
	probCorrect = evalprobs(sub2ind(size(evalprobs), labels(:, 1), (1:size(labels, 1))'));
	loss = gather(-log(probCorrect));
elseif nargin > 6 && ~isempty(score_dist)
	% Compute asymmetric cross-entropy with score_dist.
	loss = sum(score_dist .* log((score_dist + eps) ./ (probs + eps)));
	% Dummy value.
	probCorrect = loss;
elseif nargout > 1
	% Dummy values.
	probCorrect = ones(1, B);
	loss = gather(-log(probCorrect));
end

if nargin > 4 && ~isempty(multipliers)
	loss = loss .* multipliers;
	loss(isnan(loss)) = 0;
    probs(isnan(probs)) = 0;
end

end
