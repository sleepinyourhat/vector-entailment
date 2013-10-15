function data = InitializeWordFeatures (data, theta, thetaDecoder)
% Populate the word features from a V matrix into a set of trees.

[classifierMatrices, classifierMatrix, classifierBias, ...
    classifierParameters, wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias] = stack2param(theta, thetaDecoder);

for datai = 1:length(data)
   data(datai).leftTree.updateFeatures(wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias);  
   data(datai).rightTree.updateFeatures(wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias);  
end

end