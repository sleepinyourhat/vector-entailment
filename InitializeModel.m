% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta, thetaDecoder, separateWordFeatures ] = InitializeModel(wordMap, hyperParams)
% Initialize the learned parameters of the model. 

vocabLength = size(wordMap, 1);
DIM = hyperParams.dim;
EMBDIM = hyperParams.embeddingDim;
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
classifierBias = zeros(PENULT, 1);

if hyperParams.useThirdOrder
    compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* OFFSET - SCALE;
else
    compositionMatrices = zeros(0, 0, 0, NUMCOMP);
end
compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* OFFSET - SCALE;
for i = 1:NUMCOMP
  compositionMatrix(:, :, i) = compositionMatrix(:, :, i) + [eye(DIM) eye(DIM)];
end

compositionBias = zeros(DIM, NUMCOMP);

classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* OFFSET - SCALE;
classifierExtraBias = zeros(PENULT, TOPD - 1);

embeddingTransformMatrix = rand(DIM, EMBDIM, NUMTRANS) .* OFFSET - SCALE;
embeddingTransformBias = zeros(DIM, NUMTRANS);

if NUMTRANS > 0
  embeddingTransformMatrix(:, :, 1) = embeddingTransformMatrix(:, :, 1) ./ 2 + TiledEye(DIM, EMBDIM) ./ 2;
end

if hyperParams.loadWords
   Log(hyperParams.statlog, 'Loading the vocabulary.')
   wordFeatures = InitializeVocabFromFile(wordMap, hyperParams.vocabPath, SCALE);
   if ~hyperParams.trainWords
       Log(hyperParams.statlog, 'Warning: Word vectors are randomly initialized and not trained.');     
   end
else 
    % Randomly initialize the words
    wordFeatures = rand(vocabLength, EMBDIM) .* OFFSET - SCALE;
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

