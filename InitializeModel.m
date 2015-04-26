% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta, thetaDecoder, separateWordFeatures ] = InitializeModel(wordMap, hyperParams)
% Initialize the learned parameters of the model. 

assert(~(hyperParams.lstm && hyperParams.useThirdOrderComposition))
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
softmaxMatrix = InitializeNNLayer(PENULT, sum(hyperParams.numRelations), 1, 1);

mergeMatrix = InitializeNNLayer(DIM * 2, PENULT, 1, hyperParams.NNinitType);

% Randomly initialize tensor parameters
if ~hyperParams.sentenceClassificationMode
    if hyperParams.useThirdOrderMerge
        mergeMatrices = InitializeNTNLayer(DIM, PENULT, hyperParams.NTNinitType) .* hyperParams.tensorScale;
        mergeMatrix = mergeMatrix .* (1 - hyperParams.tensorScale);
    else
        mergeMatrices = zeros(0, 0, PENULT);
    end
else
    mergeMatrices = [];
    mergeMatrix = [];
end

if hyperParams.lstm
    compositionMatrix = InitializeLSTMLayer(DIM, NUMCOMP, hyperParams.LSTMinitType);
else
    compositionMatrix = InitializeNNLayer(DIM * 2, DIM, NUMCOMP, hyperParams.NNinitType);
end
  
if hyperParams.eyeScale > 0 && ~hyperParams.lstm
    for i = 1:NUMCOMP
        compositionMatrix(:, end - (2 * DIM) + 1:end, i) = compositionMatrix(:, end - (2 * DIM) + 1:end, i) .* (1 - hyperParams.eyeScale) + [eye(DIM) eye(DIM)] .* hyperParams.eyeScale;
    end
end

if hyperParams.useThirdOrderComposition && ~hyperParams.useLattices
    if hyperParams.tensorScale > 0
        compositionMatrices = InitializeNTNLayer(DIM, DIM, hyperParams.NTNinitType) .* hyperParams.tensorScale;
        compositionMatrix = compositionMatrix .* (1 - hyperParams.tensorScale);
    else
        compositionMatrices = InitializeNTNLayer(DIM, DIM, hyperParams.NTNinitType);
    end
    scoringVector = [];
elseif hyperParams.useLattices
    % To keep stacking and unstacking simple, we overload this parameter name for the 
    % connection chosing layer in the lattice model.

    % This is not a proper NN layer - just a filter that will be .*'d with a clump of features and summed.
    compositionMatrices = InitializeNNLayer((2 * hyperParams.latticeConnectionContextWidth * DIM) + 3, hyperParams.latticeConnectionHiddenDim, 1, hyperParams.NNinitType, 0);
    scoringVector = InitializeNNLayer(hyperParams.latticeConnectionHiddenDim, 1, 1, hyperParams.NNinitType);
else
    compositionMatrices = [];
    scoringVector = [];
end

classifierExtraMatrix = InitializeNNLayer(PENULT, PENULT, TOPD - 1, hyperParams.NNinitType);

  
if NUMTRANS > 0
    assert(NUMTRANS == 1, 'Currently, we do not support more than one embedding transform layer.');
    embeddingTransformMatrix = InitializeNNLayer(EMBDIM, DIM, NUMTRANS, hyperParams.NNinitType);
else
    embeddingTransformMatrix = [];
end
  
if hyperParams.loadWords
    Log(hyperParams.statlog, 'Loading the vocabulary.')
    wordFeatures = InitializeVocabFromFile(wordMap, hyperParams.vocabPath);
else 
    % Randomly initialize the words
    wordFeatures = normrnd(0, 1, EMBDIM, vocabLength);
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
[ theta, thetaDecoder ] = param2stack(mergeMatrices, mergeMatrix, ...
    softmaxMatrix, wordFeatures, compositionMatrices, ...
    compositionMatrix, scoringVector, classifierExtraMatrix, embeddingTransformMatrix);

end

