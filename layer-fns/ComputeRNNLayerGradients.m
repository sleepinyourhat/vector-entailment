% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, deltaLeft, deltaRight] = ...
      ComputeRNNLayerGradients(a, b, matrix, delta, ...
                                  nonlinearityDeriv, innerOutput)

in = [ones(1, size(a, 2)); a; b];

if nargin < 7
    innerOutput = matrix * in;
end

% Compute the gradients and deltas for an RNN layer for a given example
NLDeriv = nonlinearityDeriv(innerOutput);
delta = NLDeriv .* delta;

% Compute the matrix gradients
% TODO: Vectorize! (Tricky so far...)
matrixGradients = zeros(size(matrix, 1), size(matrix, 2), size(a, 2));
for b = 1:size(a, 2)
	matrixGradients(:, :, b) = delta(:, b) * in(:, b)';
end

if nargout > 2
	% Calculate deltas to pass down
	deltaDown = (matrix(:, 2:end)' * delta);
	deltaLeft = deltaDown(1:length(a), :);
	deltaRight = deltaDown(length(a)+1:length(deltaDown), :);
end

% TODO: Is summing happening here?



end
