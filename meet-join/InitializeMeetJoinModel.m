% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta, thetaDecoder ] = InitializeMeetJoinModel(vocabLength, hyperParams)
% Initialize the learned parameters of the model. 

DIM = hyperParams.dim;
PENULT = hyperParams.penultDim;
TOPD = hyperParams.topDepth;
NUMCOMP = 2; % Fixed: meet and join

% Randomly initialize softmax layer
classifierParameters = rand(vocabLength, PENULT + 1) .* .02 - .01;

% Randomly initialize tensor parameters
if hyperParams.useThirdOrder
    compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* .02 - .01;
else
    compositionMatrices = zeros(0, 0, 0, NUMCOMP) .* .02 - .01;
end
compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* .02 - .01; 
compositionBias = rand(DIM, NUMCOMP) .* .02 - .01;

classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* .02 - .01;
classifierExtraBias = rand(PENULT, TOPD - 1) .* .02 - .01;

% Randomly initialize the words
wordFeatures = rand(vocabLength, DIM) .* .02 - .01;

[theta, thetaDecoder] = param2stack(classifierParameters, wordFeatures, ...
    compositionMatrices, compositionMatrix, compositionBias, ... 
    classifierExtraMatrix, classifierExtraBias);

end

