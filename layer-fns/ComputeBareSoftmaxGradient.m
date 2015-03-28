% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ softmaxMatrixGradients, softmaxDelta ] = ...
    ComputeSoftmaxGradient(softmaxMatrix, probs, deltas, inputFeatures)
% Compute the gradient for the softmax layer parameters without assuming any loss function
% or a one-hot target distribution, and the deltas to pass down.

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

B = size(inputFeatures, 2);
inDim = size(inputFeatures, 1);
outDim = size(probs, 1);
in = [ones(1, B); inputFeatures];

% TODO: Save these between forward and backward passes
z = softmaxMatrix * in;
unnormedProbs = exp(z);

% TODO: Vectorize more?
% This is dProb / dZ
internalGradients = zeros(outDim, outDim, B);
for ii = 1:outDim
    for jj = 1:outDim
        internalGradients(ii, jj, :) = probs(ii) .* ((ones(1, B) * (ii == jj)) - probs(jj));
    end
end

% Transpose and multiply.
deltaZ = zeros(outDim, B);
for b = 1:B
    % TODO: Vectorize.
    deltaZ(:, b) = permute(internalGradients(:, :, b), [2, 1, 3]) * deltas(:, b);
end

% Compute the matrix gradients
softmaxMatrixGradients = (deltaZ * in');
softmaxDelta = (softmaxMatrix(:, 2:end)' * deltaZ);

end
