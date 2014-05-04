% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta, thetaDecoder ] = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams)
% Re-initializes the composition layer after pretraining, assuming that
% regularization during pretraining has pushed its parametrs to near zero. 

[classifierMatrices, classifierMatrix, classifierBias, ...
    classifierParameters, wordFeatures, ~,...
    ~, ~, classifierExtraMatrix, ...
    classifierExtraBias] = stack2param(theta, thetaDecoder);

if ~hyperParams.untied
    NUMCOMP = 1;
else
    NUMCOMP = 3;
end

DIM = hyperParams.dim;
PENULT_DIM = hyperParams.penultDim;

if hyperParams.useThirdOrder
    compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* .02 - .01;
else
    compositionMatrices = zeros(0, 0, 0, NUMCOMP) .* .02 - .01;
end
compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* .02 - .01;
compositionBias = rand(DIM, NUMCOMP) .* .02 - .01;

[theta, thetaDecoder] = param2stack(classifierMatrices, classifierMatrix, ...
    classifierBias, classifierParameters, wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias, classifierExtraMatrix, ...
    classifierExtraBias);
end

