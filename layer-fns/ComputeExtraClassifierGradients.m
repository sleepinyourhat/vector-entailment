% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixStackGradients, deltaDown ] = ...
          ComputeExtraClassifierGradients(...
          	matrixStack, deltaDown, inputs, classNLDeriv, gpu)
% Compute gradients for the middle NN layers of the classifier. This will 
% only do non-trivial work if stackSize is greater than 0.

if length(matrixStack) == 0
	 matrixStackGradients = [];
	 return
end

[ outDim, ~, stackSize ] = size(matrixStack);
[ inDim, B, ~ ] = size(inputs);

matrixStackGradients = matrixStack .* 0;  % TODO: fZeros?

% We only support different input and output dimensionalities 
% if the extra layers aren't used. Otherwise, the storage used here
% won't work.
assert(inDim == outDim || stackSize < 2, 'Inconsistent dimensions.');

for layer = stackSize:-1:1
    NLDeriv = classNLDeriv([], inputs(:, :, layer + 1));
    deltaDown = NLDeriv .* deltaDown;

    % Calculate matrix gradients
    matrixStackGradients(:, :, layer) = deltaDown * [ones([1, size(inputs(:, :, layer), 2)], 'like', inputs(:, :, layer)); inputs(:, :, layer)]';

    % Calculate deltas to pass down
    deltaDown = matrixStack(:, 2:end, layer)' * deltaDown;
end

end
