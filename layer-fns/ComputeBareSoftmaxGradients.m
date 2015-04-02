% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, deltasDown ] = ...
    ComputeBareSoftmaxGradients(matrix, probs, deltas, in)
% Compute the gradient for the softmax layer parameters using incoming
% deltas rather than log loss and a class label vector.

% TODO: (eventually maybe) add support for multiple relation classes, 
% as in the other two Softmax functions.

B = size(in, 2);
outDim = size(probs, 1);
inPadded = [ones(1, B); in];

% TODO: Save these between forward and backward passes
z = matrix * inPadded;

% Compute dProb / dZ
% TODO: Vectorize more?
zGradients = zeros(outDim, outDim, B);
for ii = 1:outDim
    for jj = 1:outDim
        zGradients(ii, jj, :) = probs(ii, :) .* ((ones(1, B) * (ii == jj)) - probs(jj, :));
    end
end

% Transpose and multiply.
% TODO: Vectorize
deltaZ = zeros(outDim, B);
for b = 1:B
    deltaZ(:, b) = zGradients(:, :, b)' * deltas(:, b);
end
deltasDown = (matrix(:, 2:end)' * deltaZ);

% Compute the matrix gradients
matrixGradients = deltaZ * inPadded';


end
