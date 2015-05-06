% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function innerTensorLayerOutput = ComputeInnerTensorLayer(l, r, matrices, matrix)
% Run an NTN layer as in forward propagation, not including the nonlinearity.

outDim = size(matrix, 1);
B = size(l, 2);

% Apply third-order tensor
% NOTE: Sadly, there doesn't seem to be a good MATLAB primitive for this.
% TODO: fZeros
innerTensorLayerOutput = zeros(outDim, B);
for outi = 1:outDim
    innerTensorLayerOutput(outi, :) = dot((matrices(:, :, outi) * r), l);
end

% Apply matrix.
innerTensorLayerOutput = innerTensorLayerOutput + matrix * padarray([l; r], 1, 1, 'pre');

end
