function [ theta, thetaDecoder ] = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams)
% Re-initializes the composition layer after pretraining. 

[classifierMatrices, classifierMatrix, classifierBias, ...
    classifierParameters, wordFeatures, ~,...
    ~, ~, classifierExtraMatrix, ...
    classifierExtraBias] = stack2param(theta, thetaDecoder);

DIM = hyperParams.dim;
PENULT_DIM = hyperParams.penultDim;

compositionMatrices = rand(DIM , DIM, DIM) .* .02 - .01;
compositionMatrix = rand(DIM, DIM * 2) .* .02 - .01;
compositionBias = rand(DIM, 1) .* .02 - .01;

[theta, thetaDecoder] = param2stack(classifierMatrices, classifierMatrix, ...
    classifierBias, classifierParameters, wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias, classifierExtraMatrix, ...
    classifierExtraBias);
end

