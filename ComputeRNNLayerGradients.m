% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, biasGradients, ...
          deltaLeft, deltaRight] = ...
      ComputeRNNLayerGradients(a, b, matrix, bias, delta, ...
                                  nonlinearityDeriv, innerOutput)

if nargin < 7
    innerOutput = matrix * [a;b] + bias;
end
                    
% Compute the gradients and deltas for an RNN layer for a given example
NLDeriv = nonlinearityDeriv(innerOutput);

% Compute the matrix gradients
matrixGradients = (delta * [a;b]');

% Calculate bias gradients
biasGradients = delta;

if nargout > 2
	% Calculate deltas to pass down
	thirdTerm = matrix';
	deltaDown = (thirdTerm * (biasGradients .* NLDeriv));
	deltaLeft = deltaDown(1:length(a));
	deltaRight = deltaDown(length(a)+1:2*length(a));
end

end