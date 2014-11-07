% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta, thetaDecoder, separateWordFeatures ] = InitializeModel(wordMap, hyperParams)
% Initialize the learned parameters of the model. 

vocabLength = size(wordMap, 1);
DIM = hyperParams.dim;
PENULT = hyperParams.penultDim;
TOPD = hyperParams.topDepth;
NUMTRANS = hyperParams.embeddingTransformDepth;
if hyperParams.useSumming
    NUMCOMP = 0;
elseif ~hyperParams.untied
    NUMCOMP = 1;
else
    NUMCOMP = 3;
end

SCALE = hyperParams.initScale;
EYESCALE = hyperParams.eyeScale;
OFFSET = 2 * hyperParams.initScale;

% Randomly initialize softmax layer
classifierParameters = rand(sum(hyperParams.numRelations), PENULT + 1) .* OFFSET - SCALE;

% Randomly initialize tensor parameters
if hyperParams.useThirdOrderComparison
    classifierMatrices = rand(DIM, DIM, PENULT) .* OFFSET - SCALE;
else
    classifierMatrices = rand(0, 0, PENULT);
end
classifierMatrix = rand(PENULT, DIM * 2) .* OFFSET - SCALE;
classifierBias = rand(PENULT, 1) .* OFFSET - SCALE;
if hyperParams.useThirdOrder
    compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* OFFSET - SCALE;
else
    compositionMatrices = zeros(0, 0, 0, NUMCOMP);
end
compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* OFFSET - SCALE;
for i = 1:NUMCOMP
  compositionMatrix(:, :, i) = compositionMatrix(:, :, i) + [eye(DIM) eye(DIM)];
end

compositionBias = rand(DIM, NUMCOMP) .* OFFSET - SCALE;

classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* OFFSET - SCALE;
classifierExtraBias = rand(PENULT, TOPD - 1) .* OFFSET - SCALE;

embeddingTransformMatrix = rand(DIM, DIM, NUMTRANS) .* OFFSET - SCALE;
embeddingTransformBias = rand(DIM, NUMTRANS) .* OFFSET - SCALE;
for matrixDepth = 1:NUMTRANS
    embeddingTransformMatrix(:, :, matrixDepth) = ...
        embeddingTransformMatrix(:, :, matrixDepth) + eye(DIM);
end

if NUMTRANS > 0
  embeddingTransformMatrix(:, :, 1) = embeddingTransformMatrix(:, :, 1) .* EYESCALE;
end

if hyperParams.loadWords
   Log(hyperParams.statlog, 'Loading the vocabulary.')
   wordFeatures = InitializeVocabFromFile(wordMap);
   if ~hyperParams.trainWords
       Log(hyperParams.statlog, 'Warning: Word vectors are randomly initialized and not trained.');     
   end
else 
    % Randomly initialize the words
    wordFeatures = rand(vocabLength, DIM) .* OFFSET - SCALE;
end

if ~hyperParams.trainWords || hyperParams.fastEmbed
    % Move the initialized word features into separateWordFeatures
    separateWordFeatures = wordFeatures;
    wordFeatures = [];
else
    separateWordFeatures = [];
end

[theta, thetaDecoder] = param2stack(classifierMatrices, classifierMatrix, ...
    classifierBias, classifierParameters, wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias, classifierExtraMatrix, ...
    classifierExtraBias, embeddingTransformMatrix, embeddingTransformBias);

end

