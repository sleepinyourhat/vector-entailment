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

assert(DIM == EMBDIM || NUMTRANS > 0, ...
  'If embeddingDim is not equal to dim, a transform layer must be used.');

% Randomly initialize softmax layer
SCALE = sqrt(6 / (PENULT + sum(hyperParams.numRelations)));
classifierParameters = rand(sum(hyperParams.numRelations), PENULT + 1) .* (2 * SCALE) - SCALE;

% Randomly initialize tensor parameters
if hyperParams.useThirdOrderComparison
    SCALE = sqrt(6 / (PENULT + 2 * DIM)) * .1;
    classifierMatrices = rand(DIM, DIM, PENULT) .* (2 * SCALE) - SCALE;
else
    classifierMatrices = rand(0, 0, PENULT);
end
SCALE = sqrt(6 / (PENULT + 2 * DIM));
classifierMatrix = rand(PENULT, DIM * 2) .* (2 * SCALE) - SCALE;
classifierBias = zeros(PENULT, 1);

if hyperParams.useThirdOrder
    SCALE = sqrt(6 / (3 * DIM)) * .1;
    compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* (2 * SCALE) - SCALE;
else
    compositionMatrices = zeros(0, 0, 0, NUMCOMP);
end
SCALE = sqrt(6 / (3 * DIM));
compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* (2 * SCALE) - SCALE;
for i = 1:NUMCOMP
  compositionMatrix(:, :, i) = compositionMatrix(:, :, i) ./ 2 + [eye(DIM) eye(DIM)] ./ 2;
end
compositionBias = zeros(DIM, NUMCOMP);

SCALE = sqrt(6 / (2 * PENULT));
classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* (2 * SCALE) - SCALE;
classifierExtraBias = zeros(PENULT, TOPD - 1);

SCALE = sqrt(6 / (EMBDIM + DIM));
embeddingTransformMatrix = rand(DIM, EMBDIM, NUMTRANS) .* (2 * SCALE) - SCALE;
embeddingTransformBias = zeros(DIM, NUMTRANS);

if NUMTRANS > 0
  embeddingTransformMatrix(:, :, 1) = embeddingTransformMatrix(:, :, 1) ./ 2 + TiledEye(DIM, EMBDIM) ./ 2;
end

if hyperParams.loadWords
   Log(hyperParams.statlog, 'Loading the vocabulary.')
   wordFeatures = InitializeVocabFromFile(wordMap, hyperParams.vocabPath);
   if ~hyperParams.trainWords
       Log(hyperParams.statlog, 'Warning: Word vectors are randomly initialized and not trained.');     
   end
else 
    % Randomly initialize the words
    SCALE = .5;
    wordFeatures = rand(vocabLength, EMBDIM) .* (2 * SCALE) - SCALE;
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

