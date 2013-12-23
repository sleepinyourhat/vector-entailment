function [ theta, thetaDecoder ] = InitializeModel(vocabLength, hyperParams)
% Initialize the learned parameters of the model. 

DIM = hyperParams.dim;
PENULT = hyperParams.penultDim;
TOPD = hyperParams.topDepth;
if ~hyperParams.untied
    NUMCOMP = 1;
else
    NUMCOMP = 3;
end

% Randomly initialize softmax layer
classifierParameters = rand(hyperParams.numRelations, PENULT + 1) .* .02 - .01;

% Randomly initialize tensor parameters
classifierMatrices = rand(DIM , DIM, PENULT) .* .02 - .01;
classifierMatrix = rand(PENULT, DIM * 2) .* .02 - .01;
classifierBias = rand(PENULT, 1) .* .02 - .01;
compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* .02 - .01;
compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* .02 - .01;
compositionBias = rand(DIM, NUMCOMP) .* .02 - .01;

classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* .02 - .01;
classifierExtraBias = rand(PENULT, TOPD - 1) .* .02 - .01;

% Randomly initialize the words
wordFeatures = rand(vocabLength, DIM) .* .02 - .01;

[theta, thetaDecoder] = param2stack(classifierMatrices, classifierMatrix, ...
    classifierBias, classifierParameters, wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias, classifierExtraMatrix, ...
    classifierExtraBias);

end

