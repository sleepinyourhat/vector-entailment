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

matrixStackGradients = zeros(outDim, inDim + 1, stackSize, B);
biasStackGradients = zeros(outDim, stackSize, B);

% We only support different input and output dimensionalities 
% if the extra layers aren't used. Otherwise, the storage used here
% won't work.
assert(inDim == outDim || stackSize < 2, 'Inconsistent dimensions.');


for layer = stackSize:-1:1
    NLDeriv = classNLDeriv(innerOutputs(:, :, layer));
    deltaDown = NLDeriv .* deltaDown;

    % Calculate matrix gradients
    for b = 1:B
	    matrixStackGradients(:, :, layer, b) = deltaDown(:, b) * [1; inputs(:, b, layer)]';
	end

    % Calculate deltas to pass down
    deltaDown = matrixStack(:, 2:end, layer)' * deltaDown;
end
    
end
