% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, deltaDown ] = ...
          ComputeEmbeddingTransformGradients(matrix, delta, in, innerOutput, nonlinearityDeriv)

inPadded = [ones(1, size(in, 2)); in];

if nargin < 7
    innerOutput = matrix * inPadded;
end

NLDeriv = nonlinearityDeriv(innerOutput);
delta = NLDeriv .* delta;

% Compute the matrix gradients
% TODO: Vectorize! (Tricky so far...)
matrixGradients = zeros(size(matrix, 1), size(matrix, 2), size(in, 2));
for b = 1:size(in, 2)
	matrixGradients(:, :, b) = delta(:, b) * inPadded(:, b)';
end

deltaDown = matrix(:, 2:end)' * delta;

end
