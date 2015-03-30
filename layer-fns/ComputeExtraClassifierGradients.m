% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixStackGradients, deltaDown ] = ...
          ComputeExtraClassifierGradients(...
          	matrixStack, deltaDown, inputs, innerOutputs, classNLDeriv)
% Compute gradients for the middle NN layers of the classifier. This will 
% only do non-trivial work if STACKSIZE is greater than 0.

if length(innerOutputs) == 0 || length(matrixStack) == 0
	matrixStackGradients = [];
	return
end

INDIM = size(matrixStack, 2);
OUTDIM = size(matrixStack, 1);
STACKSIZE = size(matrixStack, 3);

matrixStackGradients = zeros(OUTDIM, INDIM + 1, STACKSIZE);
biasStackGradients = zeros(OUTDIM, STACKSIZE);

% We only support different input and output dimensionalities 
% if there is only one layer. Otherwise, the storage used here
% won't work.
assert(INDIM == OUTDIM || STACKSIZE < 2, 'Inconsistent dimensions.');

for layer = STACKSIZE:-1:1
    NLDeriv = classNLDeriv(innerOutputs(:, layer));

    deltaDown = NLDeriv .* deltaDown;

    % Calculate matrix gradients
    matrixStackGradients(:, :, layer) = deltaDown * [1; inputs(:, layer)]';

    % Calculate deltas to pass down
    deltaDown = matrixStack(:, 2:end, layer)' * deltaDown;
end
    
end
