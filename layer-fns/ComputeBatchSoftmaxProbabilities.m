% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ relationProbs, objective ] = ComputeBatchSoftmaxProbabilities(inVector, softmaxMatrix,  trueRelations)
% Run the softmax classifier layer forward on a batch of examples.

% TODO: Add support for relation ranges ang merge this with ComputeSoftmaxProbabilities.

% Add intercept term
in = [ones(1, size(inVector, 2)); inVector];

unNormedRelationProbs = exp(softmaxMatrix * in);
partitions = sum(unNormedRelationProbs);
relationProbs = bsxfun(@rdivide, unNormedRelationProbs, partitions);

% If a correct class vector is provided, compute the objective function value.
if nargin > 2
	% Pad with ones to allow for zeros in trueRelation, which won't contribute to cost.
	evalRelationProbs = [ones(1, size(inVector, 2)); relationProbs];
	trueRelations = trueRelations + 1;
	objective = -log(evalRelationProbs(sub2ind(size(evalRelationProbs), trueRelations(:), (1:length(trueRelations))')));
else
	objective = 0;
end

end
