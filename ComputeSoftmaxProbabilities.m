function relationProbs = ComputeSoftmaxProbabilities(inVector, classifierParameters)
% Run the softmax classifier.

unNormedRelationProbs = exp(classifierParameters * inVector);

partition = sum(unNormedRelationProbs);
relationProbs = unNormedRelationProbs / partition;

end