% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta, thetaDecoder, separateWordFeatures ] = InitializeModel(wordMap, hyperParams)
% Initialize the learned parameters of the model. 

assert(~(hyperParams.lstm && hyperParams.useThirdOrder))
assert(~(hyperParams.lstm && hyperParams.eyeScale))
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

% Randomly initialize softmax layer
classifierParameters = [zeros(sum(hyperParams.numRelations), 1), ...
                        rand(sum(hyperParams.numRelations), PENULT) .* 0.002 - 0.001];

scale = 1 / sqrt(DIM);
classifierMatrix = rand(PENULT, DIM * 2) .* (2 * scale) - scale;
classifierBias = zeros(PENULT, 1);

% Randomly initialize tensor parameters
if hyperParams.useThirdOrderComparison
    scale = 2 * hyperParams.tensorScale / sqrt(DIM);
    classifierMatrices = rand(DIM, DIM, PENULT) .* (2 * scale) - scale;
    classifierMatrix = classifierMatrix .* (1 - hyperParams.tensorScale);
else
    classifierMatrices = rand(0, 0, PENULT);
end

if hyperParams.lstm
  scale = 1 / sqrt(DIM);
  compositionMatrix = rand(DIM * 4, DIM * 2 + 1, NUMCOMP) .* (2 * scale) - scale;
  compositionMatrix(:, 1) = 3 * scale;
else
  scale = 1 / sqrt(2 * DIM);
  compositionMatrix = rand(DIM, DIM * 2, NUMCOMP) .* (2 * scale) - scale;
end
  
if hyperParams.eyeScale > 0 && ~hyperParams.lstm
  for i = 1:NUMCOMP
    compositionMatrix(:, :, i) = compositionMatrix(:, :, i) .* (1 - hyperParams.eyeScale) + [eye(DIM) eye(DIM)] .* hyperParams.eyeScale;
  end
end

if hyperParams.useThirdOrder
    scale = hyperParams.tensorScale / ( 6 * sqrt(DIM) );
    compositionMatrices = rand(DIM, DIM, DIM, NUMCOMP) .* (2 * scale) - scale;
    compositionMatrix = compositionMatrix .* (1 - hyperParams.tensorScale);
else
    compositionMatrices = [];
end

if ~hyperParams.lstm
  compositionBias = zeros(DIM, NUMCOMP);
else
  compositionBias = [];
end

scale = 1/sqrt(PENULT);
classifierExtraMatrix = rand(PENULT, PENULT, TOPD - 1) .* (2 * scale) - scale;
classifierExtraBias = zeros(PENULT, TOPD - 1);

scale = 1/sqrt(EMBDIM);
embeddingTransformMatrix = rand(DIM, EMBDIM, NUMTRANS) .* (2 * scale) - scale;
embeddingTransformBias = zeros(DIM, NUMTRANS);

wordScale = 2/sqrt(EMBDIM);

if hyperParams.loadWords
   Log(hyperParams.statlog, 'Loading the vocabulary.')
   wordFeatures = InitializeVocabFromFile(wordMap, hyperParams.vocabPath, wordScale);
else 
    % Randomly initialize the words
    wordFeatures = rand(vocabLength, EMBDIM) .*   (2 * wordScale) - wordScale;
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

