function tensorLayerOutput = ComputeTensorLayer(a, b, classifierMatrices, classifierMatrix, classifierBias)
% function tensorLayerOutput = ComputeTensorLayer(a, b, classifierMatrices, classifierMatrix, classifierBias)
% From Socher et al ICML 13

tensorLayerOutput = Sigmoid(ComputeInnerTensorLayer(a, b, ...
    classifierMatrices, classifierMatrix, classifierBias));

end