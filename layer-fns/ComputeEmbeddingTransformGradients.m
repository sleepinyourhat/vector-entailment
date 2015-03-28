% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, delta ] = ...
          ComputeExtraClassifierGradients(matrix, ...
              delta, inputs, innerOutputs, classNLDeriv)

INDIM = size(matrix, 2);
OUTDIM = size(matrix, 1);

matrixGradients = zeros(OUTDIM, INDIM + 1);

NLDeriv = classNLDeriv(innerOutputs);

delta = delta .* NLDeriv;

% Calculate matrix gradients
matrixGradients = delta * [ones(size(a, 2)); inputs]';

% Calculate deltas to pass down
delta = (matrix(:, 2:end)' * delta);

end
