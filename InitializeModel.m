% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta, thetaDecoder, separateWordFeatures ] = InitializeModel(wordMap, hyperParams)
% Initialize the learned parameters of the model. 

assert(~(hyperParams.lstm && hyperParams.useThirdOrder))
assert(~(hyperParams.lstm && hyperParams.useEyes))
assert(~(hyperParams.lstm && hyperParams.useTrees))

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
    % Partial support for syntactic untying.
    NUMCOMP = 3;
end

SCALE = 0.05;
TSCALE = hyperParams.tensorScale * SCALE;

% Randomly initialize softmax layer
classifierParameters = [zeros(sum(hyperParams.numRelations), 1), ...
                        rand(sum(hyperParams.numRelations), PENULT) .* (2 * SCALE) - SCALE];

% Randomly initialize tensor parameters
if hyperParams.useThirdOrderComparison
    classifierMatrices = rand(DIM, DIM, PENULT) .* (2 * TSCALE) - TSCALE;
else
    classifierMatrices = rand(0, 0, PENULT);
end

classifierMatrix = rand(PENULT, DIM * 2) .* (2 * SCALE) - SCALE;
classifierBias = zeros(PENULT, 1);

if hyperParams.useThirdOrder
    compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* (2 * TSCALE) - TSCALE;
else
    compositionMatrices = [];
end

if hyperParams.lstm
  compositionMatrix = rand(DIM * 4, DIM * 2 + 1, NUMCOMP) .* (2 * SCALE) - SCALE;
  compositionMatrix(DIM + 1:2 * DIM, 1) = 10 * SCALE;
  compositionMatrix(2 * DIM + 1:3 * DIM, 1) = -10 * SCALE;
else
  compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* (2 * SCALE) - SCALE;
end
  

if hyperParams.useEyes
  for i = 1:NUMCOMP
    compositionMatrix(:, :, i) = compositionMatrix(:, :, i) + [eye(DIM) eye(DIM)];
  end
end

if ~hyperParams.lstm
  compositionBias = zeros(DIM, NUMCOMP);
else
  compositionBias = [];
end

classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* (2 * SCALE) - SCALE;
classifierExtraBias = zeros(PENULT, TOPD - 1);

embeddingTransformMatrix = rand(DIM, EMBDIM, NUMTRANS) .* (2 * SCALE) - SCALE;
embeddingTransformBias = zeros(DIM, NUMTRANS);

if hyperParams.loadWords
   Log(hyperParams.statlog, 'Loading the vocabulary.')
   wordFeatures = InitializeVocabFromFile(wordMap, hyperParams.vocabPath, hyperParams.wordScale);
else 
    % Randomly initialize the words
    wordFeatures = rand(vocabLength, EMBDIM) .*   (2 * hyperParams.wordScale) - hyperParams.wordScale;
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

% Pack up the parameters.
[theta, thetaDecoder] = param2stack(classifierMatrices, classifierMatrix, ...
    classifierBias, classifierParameters, wordFeatures, compositionMatrices, ...
    compositionMatrix, compositionBias, classifierExtraMatrix, ...
    classifierExtraBias, embeddingTransformMatrix, embeddingTransformBias);

end

