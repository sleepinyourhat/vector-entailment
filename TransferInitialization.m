% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function freshModelState = TransferInitialization(freshModelState, pretrainedModelState, wordMap, pretrainedWordMap, hyperParams)
% Set up for transfer learning: copy sentence embedding parameters but not
% classifier parameters into a new model state.

% TODO: Currently tied to RMSProp

% Unpack fresh model state.
[ ~, ~, ...
    softmaxMatrix, wordFeatures, ~, ...
    ~, ~, ~, ~] ...
    = stack2param(freshModelState.theta, freshModelState.thetaDecoder);

% Unpack pretrained model state.
[ mergeMatrices, mergeMatrix, ...
    ~, pretrainedWordFeatures, connectionMatrix, ...
    compositionMatrix, scoringVector, classifierExtraMatrix, embeddingTransformMatrix] ...
    = stack2param(pretrainedModelState.theta, pretrainedModelState.thetaDecoder);
[ GmergeMatrices, GmergeMatrix, ...
    ~, GpretrainedWordFeatures, GconnectionMatrix, ...
    GcompositionMatrix, GscoringVector, GclassifierExtraMatrix, GembeddingTransformMatrix] ...
    = stack2param(pretrainedModelState.sumSqGrad, pretrainedModelState.thetaDecoder);
[ DmergeMatrices, DmergeMatrix, ...
    ~, DpretrainedWordFeatures, DconnectionMatrix, ...
    DcompositionMatrix, DscoringVector, DclassifierExtraMatrix, DembeddingTransformMatrix] ...
    = stack2param(pretrainedModelState.sumSqDelta, pretrainedModelState.thetaDecoder);

if ~isempty(wordFeatures)
	wordFeaturesInStack = true;
	workingWordFeatures = wordFeatures;
else
	workingWordFeatures = freshModelState.separateWordFeatures;
	wordFeaturesInStack = false;
end

if isempty(pretrainedWordFeatures)
	pretrainedWordFeatures = pretrainedModelState.separateWordFeatures;
end
	
wordlist = wordMap.keys();
for wordlistIndex = 1:length(wordlist)
	word = wordlist{wordlistIndex};
	if pretrainedWordMap.isKey(word)
		workingWordFeatures(:, wordMap(word)) = pretrainedWordFeatures(:, pretrainedWordMap(word));
    end
end

if hyperParams.sentenceClassificationMode
	mergeMatrices = [];
	GmergeMatrices = [];
	DmergeMatrices = [];
	mergeMatrix = [];
	GmergeMatrix = [];
	DmergeMatrix = [];
end

if ~hyperParams.restartUpdateRuleInTransfer
	% Copy over the other parts of model state, used by the update rule.
    Log(hyperParams.examplelog, 'Transferring SGD state.')

	freshModelState = pretrainedModelState;
	freshModelState.step = 1;

    freshModelState.bestTestAcc = [0 0];
    freshModelState.pass = 0;
    freshModelState.lastHundredCosts = zeros(100, 1);

	% Pack up the combination.
	% TODO: Do something more intelligent with pretrained embeddings.
	[ freshModelState.sumSqGrad ] = param2stack(GmergeMatrices, GmergeMatrix, ...
	    0 .* softmaxMatrix, 0 .* wordFeatures, GconnectionMatrix, ...
	    GcompositionMatrix, GscoringVector, GclassifierExtraMatrix, GembeddingTransformMatrix);
	[ freshModelState.sumSqDelta ] = param2stack(DmergeMatrices, DmergeMatrix, ...
	    0 .* softmaxMatrix, 0.* wordFeatures, DconnectionMatrix, ...
	    DcompositionMatrix, DscoringVector, DclassifierExtraMatrix, DembeddingTransformMatrix);
else 
	freshModelState.step = 0;
end

if ~wordFeaturesInStack
	freshModelState.separateWordFeatures = workingWordFeatures;
	freshModelState.sumSqEmbGrad = 0 .* workingWordFeatures;
	freshModelState.sumSqEmbDelta = 0 .* workingWordFeatures;
else
	wordFeatures = workingWordFeatures;
end

% Pack up the combination.
[ freshModelState.theta, freshModelState.thetaDecoder ] = param2stack(mergeMatrices, mergeMatrix, ...
    softmaxMatrix, wordFeatures, connectionMatrix, ...
    compositionMatrix, scoringVector, classifierExtraMatrix, embeddingTransformMatrix);

freshModelState
freshModelState.theta(1:5000:end)
compositionMatrix(1:10, 1:10)
GcompositionMatrix(1:10, 1:10)
DcompositionMatrix(1:10, 1:10)

end
