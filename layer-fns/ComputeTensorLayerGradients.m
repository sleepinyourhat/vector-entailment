% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matricesGradients, matrixGradients, deltaLeft, deltaRight ] = ...
      ComputeTensorLayerGradients(l, r, matrices, matrix, delta, ...
                                  nonlinearityDeriv, output)
% Compute the gradients and deltas for an RNTN layer for a given batch of examples.

outDim = size(matrix, 1);
[ inDim, B ] = size(l);

% Compute the layer output if it isn't provided.
if nargin < 7
    tensorInnerOutput = ComputeInnerTensorLayer(l, r, matrices, matrix);
    NLDeriv = nonlinearityDeriv(tensorInnerOutput);
else
    NLDeriv = nonlinearityDeriv([], output);
end

delta = delta .* NLDeriv;

matricesGradients = zeros(inDim, inDim, outDim, B);

% Calculate third order tensor gradients.
% Sadly, there doesn't seem to be an efficient vectorized option here.
inputProduct = zeros(inDim, inDim, B);
for b = 1:B
	inputProduct(:, :, b) = l(:, b) * r(:, b)';
end

matricesGradients = bsxfun(@times, permute(delta, [3, 4, 1, 2]), permute(inputProduct, [1, 2, 4, 3]));
matrixInput = [ones([1, size(l, 2)], 'like', l); l; r];
matrixGradients = delta * matrixInput';

% Compute the deltas.
innerTensorLayerMatrixL = zeros(inDim, outDim);
innerTensorLayerMatrixR = zeros(inDim, outDim);
deltaLeft = zeros(inDim, b);
deltaRight = zeros(inDim, b);
for b = 1:B
    for i = 1:outDim
    	  innerTensorLayerMatrixL(:, i) = l(:, b)' * matrices(:, :, i);
        innerTensorLayerMatrixR(:, i) = matrices(:, :, i) * r(:, b);
    end

    % TODO: Need bsxfun?
    leftBackpropMatrix = bsxfun(@plus, innerTensorLayerMatrixR(:, :), matrix(:, 2:inDim + 1)');
    rightBackpropMatrix = bsxfun(@plus, innerTensorLayerMatrixL(:, :), matrix(:, inDim + 2:2 * inDim + 1)'); 
    deltaLeft(:, b) = leftBackpropMatrix * delta(:, b);
    deltaRight(:, b) = rightBackpropMatrix * delta(:, b); 
end

end
