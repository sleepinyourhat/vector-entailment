% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, deltaDown ] = ...
    ComputeBatchSoftmaxClassificationGradient(matrix, relationProbs, trueRelations, in)
% Compute the gradient for the softmax layer parameters assuming log loss for a batch.

% TODO: Add back support for multiple relation classes

B = size(relationProbs, 2);
inPadded = [ones(1, B); in];

% Reshape relation list.
trueRelations = trueRelations(:);

% Compute a nonzero target relations vector for only those batch entries that have nonzero
% target relations.
dataPointHasLabel = trueRelations(:) ~= 0;
fullRange = 1:length(trueRelations);
filteredRange = fullRange(dataPointHasLabel);
targetRelationProbs = zeros(size(relationProbs, 1), size(relationProbs, 2));
targetRelationProbs(sub2ind(size(relationProbs), trueRelations(dataPointHasLabel), filteredRange')) = 1;

deltaDown = matrix' * (relationProbs - targetRelationProbs);

% Zero out deltas for unlabeled examples, and remove bias deltas.
deltaDown = bsxfun(@times, deltaDown(2:end, :), dataPointHasLabel');

matrixGradients = zeros(size(matrix, 1), size(matrix, 2), B);
for b = 1:B
	if dataPointHasLabel(b)
		matrixGradients(:, :, b) = -((targetRelationProbs(:, b) - relationProbs(:, b)) * inPadded(:, b)');
	end
end

end
