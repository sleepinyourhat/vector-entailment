% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function freshModelState = TransferInitialization(freshModelState, pretrainedModelState, wordMap, pretrainedWordMap, hyperParams)
% Set up for transfer learning: copy sentence embedding parameters but not
% classifier parameters into a new model state.

% Unpack fresh model state.
[ mergeMatrices, mergeMatrix, ...
    softmaxMatrix, wordFeatures, ~, ...
    ~, ~, classifierExtraMatrix, ~] ...
    = stack2param(freshModelState.theta, freshModelState.thetaDecoder);

% Unpack pretrained model state.
[ ~, ~, ...
    ~, pretrainedWordFeatures, connectionMatrix, ...
    compositionMatrix, scoringVector, pretrainedClassifierExtraMatrix, embeddingTransformMatrix] ...
    = stack2param(pretrainedModelState.theta, pretrainedModelState.thetaDecoder);

if length(pretrainedClassifierExtraMatrix) == length(classifierExtraMatrix)
	classifierExtraMatrix = pretrainedClassifierExtraMatrix;
end

if ~isempty(wordFeatures)
	wordFeaturesInStack = true;
else
	wordFeatures = freshModelState.separateWordFeatures;
	wordFeaturesInStack = false;
end

if isempty(pretrainedWordFeatures)
	pretrainedWordFeatures = pretrainedModelState.separateWordFeatures;
end
	
wordlist = wordMap.keys();
copied = 0;
for wordlistIndex = 1:length(wordlist)
	word = wordlist{wordlistIndex};
	if pretrainedWordMap.isKey(word)
		wordFeatures(:, wordMap(word)) = pretrainedWordFeatures(:, pretrainedWordMap(word));
		copied = copied + 1;
    end
end

if ~hyperParams.restartUpdateRuleInTransfer
	% Copy over the other parts of model state, used by the update rule.
    Log(hyperParams.examplelog, 'Transferring SGD state.')

	freshModelState = pretrainedModelState;
	freshModelState
	freshModelState.step = 1;

    freshModelState.bestTestAcc = [0 0];
    freshModelState.pass = 0;
    freshModelState.lastHundredCosts = zeros(100, 1);
else 
	freshModelState.step = 0;
end
	

if ~wordFeaturesInStack
	freshModelState.separateWordFeatures = wordFeatures;
	wordFeatures = [];
end

% Pack up the combination.
[ freshModelState.theta, freshModelState.thetaDecoder ] = param2stack(mergeMatrices, mergeMatrix, ...
    softmaxMatrix, wordFeatures, connectionMatrix, ...
    compositionMatrix, scoringVector, classifierExtraMatrix, embeddingTransformMatrix);

freshModelState.separateWordFeatures = pretrainedModelState.separateWordFeatures;

end
