function gradient_error = TestGradientInCombinedFeatures (i, delta, classifierMatrices, classifierParameters, combinedFeatures, trueRelation)
% gradient_error = TestGradientInCombinedFeatures (word, i, delta, classifierMatrices, classifierParameters, combinedFeatures, trueRelation)
% Find out how close a gradinent for one entry in classifierMatrices is to
% the local slope.

EPSILON = 1;

originalValue = combinedFeatures(i);
objs = [];
for polarity = -1:2:1
    combinedFeatures(i) = originalValue + polarity * EPSILON;
    tensorOutput = ComputeTensorLayer(combinedFeatures, classifierMatrices);    
    probs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters);
    objs = [objs Objective(trueRelation, probs)];
end
    
localSlope = (objs(2) - objs(1)) / (2 * EPSILON)
delta
gradient_error = (localSlope / delta)
    
% if localSlope > 0.1
%     % For histogram of error:
%     % global gradient_errors;
%     % gradient_errors = [gradient_errors gradient_error];
%     
% else
%     gradient_error = 1;
% end
    

end