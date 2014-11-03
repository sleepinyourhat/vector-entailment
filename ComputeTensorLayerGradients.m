% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matricesGradients, matrixGradients, biasGradients, ...
          deltaLeft, deltaRight] = ...
      ComputeTensorLayerGradients(a, b, matrices, matrix, bias, delta, ...
                                  nonlinearityDeriv, tensorInnerOutput)
% Compute the gradients and deltas for an RNTN layer for a given example.

% Compute the layer output if it isn't provided.
if nargin < 8
    tensorInnerOutput = ComputeInnerTensorLayer(a, b, matrices, matrix, bias);
end

tensorDeriv = nonlinearityDeriv(tensorInnerOutput);

[outDim, inDim] = size(matrix);
inDim = inDim / 2;

matricesGradients = zeros(inDim , inDim, outDim);
matrixGradients = zeros(outDim, 2 * inDim);

% Calculate third order tensor gradients
for i = 1:outDim
    matricesGradients(:,:,i) = (tensorDeriv(i) * delta(i)) .* (a * b');
end
    
% Calculate matrix gradients for tensor layer
for i = 1:outDim
    matrixGradients(i, :) = (tensorDeriv(i) * delta(i)) .* [a; b];
end

% Calculate vector gradients for tensor layer
biasGradients = (tensorDeriv .* delta);
delta = biasGradients;

innerTensorLayerMatrix = zeros(inDim, outDim);
for i = 1:outDim
    innerTensorLayerMatrix(:, i) = matrices(:,:,i) * b;
end
thirdTerm = innerTensorLayerMatrix + matrix(:, 1:inDim)';
deltaLeft = (thirdTerm * (delta .* tensorDeriv));

innerTensorLayerMatrix = zeros(inDim, outDim);
for i = 1:outDim
    innerTensorLayerMatrix(:, i) = a' * matrices(:,:,i);
end
thirdTerm = innerTensorLayerMatrix + matrix(:, inDim+1:2*inDim)';    
deltaRight = (thirdTerm * (delta .* tensorDeriv));

end