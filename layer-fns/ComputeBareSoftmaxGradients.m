% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, deltasDown ] = ...
    ComputeBareSoftmaxGradients(matrix, probs, deltas, in, gpu)
% Compute the gradient for the softmax layer parameters using incoming
% deltas rather than log loss and a class label vector.

% TODO: (eventually maybe) add support for multiple label classes, 
% as in the other two Softmax functions.

B = size(in, 2);
outDim = size(probs, 1);
inPadded = [fOnes([1, B], gpu); in];

% TODO: Save these between forward and backward passes
if ~isempty(matrix)
	z = matrix * inPadded;
else
	z = in;
end

% Compute dProb / dZ
% TODO: Vectorize more?
oneVec = ones([1, B], 'like', in);
zGradients = zeros([outDim, outDim, B], 'like', in);
for ii = 1:outDim
    for jj = 1:outDim
        zGradients(ii, jj, :) = probs(ii, :) .* ((oneVec * (ii == jj)) - probs(jj, :));
    end
end

% Transpose and multiply.
deltaZ = zeros([outDim, B], 'like', in);
for b = 1:B
    deltaZ(:, b) = zGradients(:, :, b)' * deltas(:, b);
end

if ~isempty(matrix)
	deltasDown = (matrix(:, 2:end)' * deltaZ);

	% Compute the matrix gradients
	matrixGradients = deltaZ * inPadded';
else
	deltasDown = deltaZ;
	matrixGradients = [];
end

end
