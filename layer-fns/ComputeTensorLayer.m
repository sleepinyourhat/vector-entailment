% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [tensorLayerOutput, innerTensorLayerOutput]= ComputeTensorLayer(l, r, matrices, matrix, NL)
% Run an NTN layer as in forward propagation.

innerTensorLayerOutput = ComputeInnerTensorLayer(l, r, matrices, matrix);

tensorLayerOutput = NL(innerTensorLayerOutput);

end
