function gradient_error = TestGradientInSoftmax(i, j, delta, classifierParameters, tensorOutput, trueRelation)
% TestGradientInSoftmax(testi, testj, softmaxGradients(testi,testj), classifierParameters, tensorLayerOutput, trueRelation)
% Find out how close a gradinent for one entry in classifierMatrices is to
% the local slope.

EPSILON = 0.0001;

originalValue = classifierParameters(i, j);
objs = [];
for polarity = -1:2:1
    classifierParameters(i,j) = originalValue + polarity * EPSILON;
    probs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters);
    objs = [objs Objective(trueRelation, probs)];
end

localSlope = (objs(2) - objs(1)) / (2 * EPSILON);

gradient_error = (localSlope / delta);

end
