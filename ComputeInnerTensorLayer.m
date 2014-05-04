% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function innerTensorLayerOutput = ComputeInnerTensorLayer(a, b, matrices, matrix, bias)
% Run an RNTN layer as in forward propagation, not including the
% nonlinearity.

[outDim, inDim] = size(matrix);
% inDim = inDim / 2;

innerTensorLayerOutput = zeros(outDim,1);

% Apply third-order tensor
for outi = 1:outDim
    innerTensorLayerOutput(outi) = a' * matrices(:,:,outi) * b;
end

% Apply matrix
innerTensorLayerOutput = innerTensorLayerOutput + matrix * [a; b];

% Apply bias
innerTensorLayerOutput = innerTensorLayerOutput + bias;

end