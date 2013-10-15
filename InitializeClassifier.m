function [ theta, thetaDecoder ] = InitializeClassifier(vocabLength, hyperParams)
% Initializes the learned parameters of the model, excluding the word
% representations. 

DIM = hyperParams.dim;
PENULT_DIM = hyperParams.penultDim;

% Randomly initialize softmax layer.
classifierParameters = rand(NUM_RELATIONS, PENULT_DIM + 1) .* .02 - .01;

% Randomly initialize tensor matrices.
classifierMatrices = rand(DIM , (DIM * PENULT_DIM)) .* .02 - .01;
classifierMatrix = rand(PENULT_DIM, DIM * 2) .* .02 - .01;
classifierBias = rand(PENULT_DIM, 1) .* .02 - .01;

[theta, thetaDecoder] = param2stack(classifierMatrices, classifierMatrix, ...
    classifierBias, classifierParameters, wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias);

end

