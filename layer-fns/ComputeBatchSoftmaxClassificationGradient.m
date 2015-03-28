% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [softmaxGradients, softmaxDelta] = ...
    ComputeSoftmaxGradient (hyperParams, softmaxMatrix, ...
                            relationProbs, trueRelations, mergeOutput)
% Compute the gradient for the softmax layer parameters assuming log loss, 
% and the deltas to pass down.

% Note: Relation range specifies which relations are under consideration. If 
% relationRange covers the whole space of relations suported by the parameter
% matrix (i.e., relationRange = 1:size(softmaxMatrix, 1)), then this computes
% the gradient for a single normal softmax classifier. If this is not the case, then
% columns of the matrix that aren't included in relationRange are ignored, and assumed
% to not contribute to the output distribution.

% This configuration is used to allow for one trained network to be trained using examples
% which were labeled from label sets that don't correspond exactly to the label set used
% on the test data.

% TODO: Add back support for multiple relation classes

B = size(relationProbs, 2);

in = [ones(1, B); mergeOutput];


targetRelationProbs = zeros(length(relationProbs), B);
targetRelationProbs(sub2ind(size(targetRelationProbs), trueRelations, 1:size(trueRelations, 2))) = 1;

softmaxDeltaFirstHalf = softmaxMatrix' * ...
                        (relationProbs - targetRelationProbs);
softmaxDeltaSecondHalf = hyperParams.classNLDeriv(in);
softmaxDelta = (softmaxDeltaFirstHalf .* softmaxDeltaSecondHalf);
softmaxDelta = softmaxDelta(2:hyperParams.penultDim+1, :);

softmaxGradients = zeros(size(softmaxMatrix, 1), hyperParams.penultDim + 1, B);
for relEval = 1:size(softmaxMatrix, 1)
    softmaxGradients(relEval, :, :) = -bsxfun(@times, in, (targetRelationProbs(relEval, :) - relationProbs(relEval, :)));
end

end
