% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matricesGradients, matrixGradients, ...
           deltaLeft, deltaRight ] = ...
      ComputeTensorLayerGradients(a, b, matrices, matrix, delta, ...
                                  nonlinearityDeriv, tensorInnerOutput)
% Compute the gradients and deltas for an RNTN layer for a given example.

% Compute the layer output if it isn't provided.
if nargin < 8
    tensorInnerOutput = ComputeInnerTensorLayer(a, b, matrices, matrix);
end

tensorDeriv = nonlinearityDeriv(tensorInnerOutput);

delta = delta .* tensorDeriv;

[ outDim, inDim ] = size(matrix);
inDim = (inDim - 1) / 2;

matricesGradients = zeros(inDim , inDim, outDim);
matrixGradients = zeros(outDim, 2 * inDim);

% Calculate third order tensor gradients.
% Sadly, there doesn't seem to be an efficient vectorized option here.
inputProduct = (a * b');
for i = 1:outDim
    matricesGradients(:,:,i) = delta(i) .* inputProduct;
end
    
% Calculate matrix gradients for tensor layer
matrixGradients = (delta * [ones(1, size(a, 2)); a; b]');

% Compute the deltas.
innerTensorLayerMatrixA = zeros(inDim, outDim);
innerTensorLayerMatrixB = zeros(inDim, outDim);
for i = 1:outDim
	innerTensorLayerMatrixA(:, i) = a' * matrices(:,:,i);
    innerTensorLayerMatrixB(:, i) = matrices(:,:,i) * b;
end

leftBackpropMatrix = innerTensorLayerMatrixB + matrix(:, 2:inDim + 1)';
deltaLeft = (leftBackpropMatrix * delta);

rightBackpropMatrix = innerTensorLayerMatrixA + matrix(:, inDim + 2:2 * inDim + 1)';    
deltaRight = (rightBackpropMatrix * delta);

end
