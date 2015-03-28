% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function innerTensorLayerOutput = ComputeInnerTensorLayer(a, b, matrices, matrix)
% Run an RNTN layer as in forward propagation, not including the
% nonlinearity.

[ outDim, inDim ] = size(matrix);
% inDim = inDim / 2;

innerTensorLayerOutput = zeros(outDim, 1);

% Apply third-order tensor
% NOTE: Sadly, there doesn't seem to be a good MATLAB primitive for this
for outi = 1:outDim
    innerTensorLayerOutput(outi) = a' * matrices(:,:,outi) * b;
end

% Apply matrix
innerTensorLayerOutput = innerTensorLayerOutput + matrix * [ones(1, size(a, 2)); a; b];

% TODO: Try this from Socher:
% H = bsxfun(@times,A,reshape(v,[1 1 length(v)]));
% H = sum(H ,3);

end
