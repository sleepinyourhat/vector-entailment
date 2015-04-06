% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, deltaLeft, deltaRight] = ...
      ComputeRNNLayerGradients(l, r, matrix, delta, nonlinearityDeriv, output)
% Compute the gradients and deltas for an RNN layer for a given batch.

in = [ones(1, size(l, 2)); l; r];

% TODO: Make this happen less often.
if nargin < 6
    innerOutput = matrix * in;
    NLDeriv = nonlinearityDeriv(innerOutput);
else
	NLDeriv = nonlinearityDeriv([], output);
end

% TODO: Efficient NLDeriv
delta = NLDeriv .* delta;

% Compute the matrix gradients
% TODO: Preallocate and pass in?
matrixGradients = delta * in';

if nargout > 2
	% Calculate deltas to pass down
	deltaDown = matrix(:, 2:end)' * delta;
	deltaLeft = deltaDown(1:size(l, 1), :);
	deltaRight = deltaDown(size(l, 1) + 1:size(deltaDown, 1), :);
end

end
