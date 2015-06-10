% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [tensorLayerOutput, innerTensorLayerOutput]= ComputeTensorLayer(l, r, matrices, matrix, NL)
% Run an NTN layer as in forward propagation.

% TODO: Make GPU-safe (and/or fully batch)
innerTensorLayerOutput = zeros(size(matrices, 3), size(l, 2));
for b = 1:size(l, 2)
	innerTensorLayerOutput(:, b) = ComputeInnerTensorLayer(l(:, b), r(:, b), matrices, matrix);
end

tensorLayerOutput = NL(innerTensorLayerOutput);

end
