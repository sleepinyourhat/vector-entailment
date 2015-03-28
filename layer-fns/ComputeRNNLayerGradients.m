% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, deltaLeft, deltaRight] = ...
      ComputeRNNLayerGradients(a, b, matrix, delta, ...
                                  nonlinearityDeriv, innerOutput)

if nargin < 7
    innerOutput = matrix * [ones(1, size(a, 2)); a; b];
end

% Compute the gradients and deltas for an RNN layer for a given example
NLDeriv = nonlinearityDeriv(innerOutput);
delta = NLDeriv .* delta;

% Compute the matrix gradients
matrixGradients = (delta * [ones(1, size(a, 2)); a; b]');

if nargout > 2
	% Calculate deltas to pass down
	deltaDown = (matrix(:, 2:end)' * delta);
	deltaLeft = deltaDown(1:length(a), :);
	deltaRight = deltaDown(length(a)+1:length(deltaDown), :);
end

% TODO: Is summing happening here?

end
