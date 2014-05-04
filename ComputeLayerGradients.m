% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, biasGradients, ...
          deltaLeft, deltaRight] = ...
      ComputeLayerGradients(a, b, matrix, bias, delta, ...
                                  nonlinearityDeriv, tensorInnerOutput)
                              
if nargin < 7
    tensorInnerOutput = matrix * [a;b] + bias;
end

                              
% Compute the gradients and deltas for an RNN layer for a given example.
NLDeriv = nonlinearityDeriv(tensorInnerOutput);

matrixGradients = (delta * [a;b]');

% Calculate bias gradients
biasGradients = (NLDeriv .* delta);

% Calculate deltas to pass down
thirdTerm = matrix';
deltaDown = (thirdTerm * (biasGradients .* NLDeriv));

deltaLeft = deltaDown(1:length(a));
deltaRight = deltaDown(length(a)+1:2*length(a));