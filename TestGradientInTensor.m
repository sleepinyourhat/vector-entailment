function gradient_error = TestGradientInTensor (i, grad, params, decoder, a, b, trueRelation)
% TestGradientInTensor (i, delta, feats, decoder, a, b, trueRelation)
% Find out how close a gradinent for one entry in tensor parameters is to
% the local slope.
% Use: feats = param2stack(classifierMatrices, classifierMatrix, classifierBias)

DELTA = 0.0001;

originalValue = params(i);
objs = [];
for polarity = -1:2:1
    params(i) = originalValue + polarity * DELTA;
    [classifierMatrices, classifierMatrix, classifierBias, classifierParameters] = stack2param(params, decoder);
    tensorOutput = ComputeTensorLayer(a, b, classifierMatrices, classifierMatrix, classifierBias);    
    probs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters);
    objs = [objs Objective(trueRelation, probs, params)];
end

localSlope = (objs(2) - objs(1)) / (2 * DELTA);

gradient_error = (grad / localSlope);
    
% if localSlope > 0.1
%     % For histogram of error:
%     % global gradient_errors;
%     % gradient_errors = [gradient_errors gradient_error];
%     
% else
%     gradient_error = 1;
% end
    

end