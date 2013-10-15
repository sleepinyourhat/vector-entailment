function relationProbs = ComputeSoftmaxProbabilities(inVector, classifierParameters)
% Run the softmax classifier.

% Add intercept term
input = [1; inVector];

unNormedRelationProbs = exp(classifierParameters * input);

partition = sum(unNormedRelationProbs);
relationProbs = unNormedRelationProbs / partition;

end