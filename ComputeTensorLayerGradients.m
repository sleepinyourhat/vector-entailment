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

% Calculate third order tensor gradients.
% Sadly, there doesn't seem to be an efficient vectorized option here.
inputProduct = (a * b');
for i = 1:outDim
    matricesGradients(:,:,i) = (tensorDeriv(i) * delta(i)) .* inputProduct;
end
    
% Calculate matrix gradients for tensor layer
matrixGradients = (delta * [a; b]');

% Calculate vector gradients for tensor layer
biasGradients = delta;

% Compute the deltas.
innerTensorLayerMatrixA = zeros(inDim, outDim);
innerTensorLayerMatrixB = zeros(inDim, outDim);
for i = 1:outDim
	innerTensorLayerMatrixA(:, i) = a' * matrices(:,:,i);
    innerTensorLayerMatrixB(:, i) = matrices(:,:,i) * b;
end

thirdTerm = innerTensorLayerMatrixB + matrix(:, 1:inDim)';
deltaLeft = (thirdTerm * (delta .* tensorDeriv));

thirdTerm = innerTensorLayerMatrixA + matrix(:, inDim+1:2*inDim)';    
deltaRight = (thirdTerm * (delta .* tensorDeriv));

end