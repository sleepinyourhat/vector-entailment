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

SCALE = 0.05;
TSCALE = hyperParams.tensorScale * SCALE;

% Randomly initialize softmax layer
% SCALE = 1 / sqrt(PENULT + 1);
classifierParameters = [zeros(sum(hyperParams.numRelations) + SCALE, 1), ...
                        rand(sum(hyperParams.numRelations), PENULT) .* (2 * SCALE) - SCALE];

% Randomly initialize tensor parameters
% SCALE = (1 / (2 * sqrt(DIM))) * hyperParams.initScale;
if hyperParams.useThirdOrderComparison
    classifierMatrices = rand(DIM, DIM, PENULT) .* (2 * TSCALE) - TSCALE;
else
    classifierMatrices = rand(0, 0, PENULT);
end
% SCALE = 1 / (2 * sqrt(2 * DIM));
classifierMatrix = rand(PENULT, DIM * 2) .* (2 * SCALE) - SCALE;
classifierBias = zeros(PENULT, 1);

% SCALE = (1 / (2 * sqrt(DIM))) * hyperParams.initScale;
if hyperParams.useThirdOrder
    compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* (2 * TSCALE) - TSCALE;
else
    compositionMatrices = zeros(0, 0, 0, NUMCOMP);
end
% SCALE = 1 / (2 * sqrt(2 * DIM));
compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* (2 * SCALE) - SCALE;

for i = 1:NUMCOMP
  compositionMatrix(:, :, i) = compositionMatrix(:, :, i) ./ 2 + [eye(DIM) eye(DIM)] ./ 4;
end


compositionBias = zeros(DIM, NUMCOMP);

% SCALE = 1 / sqrt(PENULT);
classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* (2 * SCALE) - SCALE;
classifierExtraBias = zeros(PENULT, TOPD - 1);

% SCALE = 1 / sqrt(EMBDIM);
embeddingTransformMatrix = rand(DIM, EMBDIM, NUMTRANS) .* (2 * SCALE) - SCALE;
embeddingTransformBias = zeros(DIM, NUMTRANS);

if hyperParams.useEyes
  if NUMTRANS > 0
    embeddingTransformMatrix(:, :, 1) = embeddingTransformMatrix(:, :, 1) ./ 2 + TiledEye(DIM, EMBDIM) ./ ((EMBDIM / DIM) * 2);
  end
end

if hyperParams.loadWords
   Log(hyperParams.statlog, 'Loading the vocabulary.')
   wordFeatures = InitializeVocabFromFile(wordMap, hyperParams.vocabPath, hyperParams.wordScale);
else 
    % Randomly initialize the words
    wordFeatures = rand(vocabLength, EMBDIM) .* (2 * hyperParams.wordScale) - hyperParams.wordScale;
    if ~hyperParams.trainWords
       Log(hyperParams.statlog, 'Warning: Word vectors are randomly initialized and not trained.');     
   end
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

