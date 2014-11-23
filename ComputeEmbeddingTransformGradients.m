% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixGradients, ...
          biasGradients, delta] = ...
          ComputeExtraClassifierGradients(matrix, bias, ...
              delta, inputs, innerOutputs, classNLDeriv)

% assert(min(innerOutputs == matrix * inputs + bias) == 1, 'ERROR!');

INDIM = size(matrix, 2);
OUTDIM = size(matrix, 1);

matrixGradients = zeros(OUTDIM, INDIM);
biasGradients = zeros(OUTDIM);

NLDeriv = classNLDeriv(innerOutputs);

delta = delta .* NLDeriv;

% Calculate matrix gradients
matrixGradients = delta * inputs';

% Calculate bias gradients
biasGradients = delta;

% Calculate deltas to pass down
delta = (matrix' * delta);

end