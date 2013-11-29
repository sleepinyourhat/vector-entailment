function innerTensorLayerOutput = ComputeInnerTensorLayer(a, b, matrices, matrix, bias)
% function tensorLayerOutput = ComputeTensorLayer(a, b, classifierMatrices, classifierMatrix, classifierBias)
% From Socher et al ICML 13

[outDim, inDim] = size(matrix);
inDim = inDim / 2;

innerTensorLayerOutput = zeros(outDim,1);

% Apply third-order tensor
for outi = 1:outDim
    % Cols = (inDim*(outi - 1))+1:(inDim*outi);
    innerTensorLayerOutput(outi) = a' * matrices(:,:,outi) * b;
end

% Apply matrix
innerTensorLayerOutput = innerTensorLayerOutput + matrix * [a; b];

% Apply bias
innerTensorLayerOutput = innerTensorLayerOutput + bias;

end