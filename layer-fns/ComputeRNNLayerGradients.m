% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, deltaLeft, deltaRight] = ...
      ComputeRNNLayerGradients(l, r, matrix, delta, ...
                                  nonlinearityDeriv, innerOutput)
% Compute the gradients and deltas for an RNN layer for a given batch.

in = [ones(1, size(l, 2)); l; r];

% TODO: Make this happen less often.
if nargin < 7
    innerOutput = matrix * in;
end

% TODO: Efficient NLDeriv
NLDeriv = nonlinearityDeriv(innerOutput);
delta = NLDeriv .* delta;

% Compute the matrix gradients
% TODO: Vectorize! (Tricky so far...)

% TODO: Preallocate and pass in?
matrixGradients = zeros(size(matrix, 1), size(matrix, 2), size(l, 2));
% VECTORIZE! Or parfor?
for b = 1:size(l, 2)
	matrixGradients(:, :, b) = delta(:, b) * in(:, b)';
end

if nargout > 2
	% Calculate deltas to pass down
	deltaDown = matrix(:, 2:end)' * delta;
	deltaLeft = deltaDown(1:size(l, 1), :);
	deltaRight = deltaDown(size(l, 1) + 1:size(deltaDown, 1), :);
end

end
