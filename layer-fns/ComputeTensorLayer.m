% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [tensorLayerOutput, innerTensorLayerOutput]= ComputeTensorLayer(a, b, matrices, matrix, bias, NL)
% Run an RNTN layer as in forward propagation.

innerTensorLayerOutput = ComputeInnerTensorLayer(a, b, matrices, matrix, bias);

tensorLayerOutput = NL(innerTensorLayerOutput);

end