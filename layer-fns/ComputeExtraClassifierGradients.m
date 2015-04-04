% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixStackGradients, deltaDown ] = ...
          ComputeExtraClassifierGradients(...
          	matrixStack, deltaDown, inputs, innerOutputs, classNLDeriv)
% Compute gradients for the middle NN layers of the classifier. This will 
% only do non-trivial work if stackSize is greater than 0.

if length(innerOutputs) == 0 || length(matrixStack) == 0
	matrixStackGradients = [];
	return
end

[ outDim, ~, stackSize ] = size(matrixStack);
[ inDim, B, ~ ] = size(inputs);

matrixStackGradients = zeros(outDim, inDim + 1, stackSize);
biasStackGradients = zeros(outDim, stackSize, B);

% We only support different input and output dimensionalities 
% if the extra layers aren't used. Otherwise, the storage used here
% won't work.
assert(inDim == outDim || stackSize < 2, 'Inconsistent dimensions.');

for layer = stackSize:-1:1
    NLDeriv = classNLDeriv(innerOutputs(:, :, layer));
    deltaDown = NLDeriv .* deltaDown;

    % Calculate matrix gradients
    matrixStackGradients(:, :, layer) = deltaDown * [ones(1, B); inputs(:, :, layer)]';

    % Calculate deltas to pass down
    deltaDown = matrixStack(:, 2:end, layer)' * deltaDown;
end

end
