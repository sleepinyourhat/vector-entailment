function innerTensorLayerOutput = ComputeInnerTensorLayer(a, b, matrices, matrix, bias)
% function tensorLayerOutput = ComputeTensorLayer(a, b, classifierMatrices, classifierMatrix, classifierBias)
% From Socher et al ICML 13

[outDim, inDim] = size(matrix);
inDim = inDim / 2;

n = outDim;
 first = zeros(n,n);
 left = a';
 right = b';

 for i = 1:n
     first(i,:) = left*matrices(:,:,i);
 end
 sec = bsxfun(@times,right,first);
 quad = squeeze(sum(sec,2));
 if size(quad,1) == 1
     quad = quad';
 end
 
 innerTensorLayerOutput = quad;

% Apply matrix
innerTensorLayerOutput = innerTensorLayerOutput + matrix * [a; b];

% Apply bias
innerTensorLayerOutput = innerTensorLayerOutput + bias;









end