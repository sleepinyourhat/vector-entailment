% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, deltaDown ] = ...
    ComputeSoftmaxClassificationGradients(matrix, probs, labels, in, hyperParams, multipliers, score_dist)
% Compute the gradient for the softmax layer parameters assuming log loss for a batch.

B = size(probs, 2);
inPadded = [ones([1, size(in, 2)], 'like', in); in];

matrixGradients = fZeros([size(matrix, 1), size(matrix, 2), B], hyperParams.gpu);

% Compute a nonzero target labels vector for only those batch entries that have nonzero
% target labels.

if ~isempty(labels)
	dataPointHasLabel = labels(:, 1) ~= 0;
	fullRange = 1:size(labels, 1);
	filteredRange = fullRange(dataPointHasLabel);

	targetprobs = fZeros(size(probs), hyperParams.gpu);

	% TODO: Speed up... oddly slow.
	targetprobs(sub2ind(size(probs), labels(dataPointHasLabel, 1), filteredRange')) = 1;
else
	assert(~isempty(score_dist), 'Must provide either labels or score_dist to get classification gradients from softmax.');
	targetprobs = score_dist;
	dataPointHasLabel = 1;
end
	

if size(labels, 2) == 2
	% Multiple class set case.

	for b = 1:B
		labelRange = hyperParams.labelRanges{labels(b, 2)};
		delta = probs(1:length(labelRange), b) - targetprobs(1:length(labelRange), b);
		if nargin > 5 && ~isempty(multipliers)
			% Scale the deltas by the multipliers (optional)
			delta = delta .* multipliers(b);
		end

		if ~isempty(matrix)
			if dataPointHasLabel(b)
				matrixGradients(labelRange, :, b) = delta * inPadded(:, b)';
			end
			deltaDown(:, b) = matrix(labelRange, :)' * delta;
		else
			deltaDown(:, b) = delta;
		end
	end
else
	if nargin > 5 && ~isempty(multipliers)
		% Scale the delatas by the multipliers (optional)		
		delta = bsxfun(@times, (probs - targetprobs), dataPointHasLabel' .* multipliers');
	else
		delta = bsxfun(@times, (probs - targetprobs), dataPointHasLabel');
	end

	if ~isempty(matrix)
		matrixGradients = delta * inPadded';
		deltaDown = matrix' * delta;
	else
		deltaDown = delta;
	end
end

if ~isempty(matrix)
	% Remove bias deltas.
	deltaDown = deltaDown(2:end, :);
end

end
