function gradient_error = TestGradientInWord (i, delta, params, paramdecoder, words, worddecoder, trueRelation)
% function gradient_error = TestGradientInWord (i, delta, params, paramdecoder, words, worddecoder, trueRelation)
% Find out how close a gradinent for one entry in two top level word parameters is to
% the local slope.
% Use: [feats, featdecoder] = param2stack(classifierMatrices, classifierMatrix, classifierBias)
%      [words, worddecoder] = param2stack(a, b);

EPSILON = 0.001;

originalValue = words(i);
objs = [];

[classifierMatrices, classifierMatrix, classifierBias, ...
    classifierParameters] = stack2param(params, paramdecoder);

for polarity = -1:2:1
    words(i) = originalValue + polarity * EPSILON;
    [a, b] = stack2param(words, worddecoder);
    tensorOutput = ComputeTensorLayer(a, b, classifierMatrices, ...
                            classifierMatrix, classifierBias);    
    probs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters);
    objs = [objs Objective(trueRelation, probs, [params; words])];
end

localSlope = (objs(2) - objs(1)) / (2 * EPSILON);

gradient_error = (localSlope / delta);
    
% if localSlope > 0.1
%     % For histogram of error:
%     % global gradient_errors;
%     % gradient_errors = [gradient_errors gradient_error];
%     
% else
%     gradient_error = 1;
% end
    

end