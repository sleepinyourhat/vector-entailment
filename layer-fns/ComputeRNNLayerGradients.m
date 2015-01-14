% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, biasGradients, ...
          deltaLeft, deltaRight] = ...
      ComputeRNNLayerGradients(a, b, matrix, bias, delta, ...
                                  nonlinearityDeriv, innerOutput)

if nargin < 7
    innerOutput = matrix * [a; b] + bias;
	% Slow diagnostic assert:
	% assert(min(innerOutput == matrix * [a;b] + bias), '!');
end

% Compute the gradients and deltas for an RNN layer for a given example
NLDeriv = nonlinearityDeriv(innerOutput);

delta = NLDeriv .* delta;

% Calculate bias gradients
biasGradients = delta;

% Compute the matrix gradients
matrixGradients = (delta * [a;b]');

if nargout > 2
	% Calculate deltas to pass down
	deltaDown = (matrix' * delta);
	deltaLeft = deltaDown(1:length(a));
	deltaRight = deltaDown(length(a)+1:length(deltaDown));
end

end