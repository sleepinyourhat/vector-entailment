% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, deltaDown ] = ...
    ComputeSoftmaxClassificationGradients(matrix, probs, labels, in, relationRange)
% Compute the gradient for the softmax layer parameters assuming log loss for a batch.

B = size(probs, 2);
inPadded = [ones(1, B); in];

if nargin < 5
	relationRange = 1:size(matrix, 1);
end

% Reshape label list.
labels = labels(:);

% Compute a nonzero target relations vector for only those batch entries that have nonzero
% target relations.
dataPointHasLabel = labels(:) ~= 0;
fullRange = 1:length(labels);
filteredRange = fullRange(dataPointHasLabel);
targetprobs = zeros(size(probs, 1), size(probs, 2));
targetprobs(sub2ind(size(probs), labels(dataPointHasLabel), filteredRange')) = 1;

deltaDown = matrix(relationRange, :)' * (probs - targetprobs);

% Zero out deltas for unlabeled examples, and remove bias deltas.
deltaDown = bsxfun(@times, deltaDown(2:end, :), dataPointHasLabel');

matrixGradients = zeros(size(matrix, 1), size(matrix, 2), B);
for b = 1:B
	if dataPointHasLabel(b)
		matrixGradients(relationRange, :, b) = -((targetprobs(:, b) - probs(:, b)) * inPadded(:, b)');
	end
end

end
