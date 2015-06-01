function [ hyperParams, options, wordMap, labelMap ] = RTE3(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, wordsource, dp, gc, adi)
% Configure RTE3 experiments.
% See Defaults.m for parameter descriptions.

[hyperParams, options] = Defaults();
hyperParams.largeVocabMode = true;
hyperParams.loadWords = true;
hyperParams.trainWords = true;

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-pen', num2str(penult), '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(wordsource),...
    '-comp', num2str(composition), ...
    '-dp', num2str(dp), '-gc', num2str(gc),  '-adi', num2str(adi)];

hyperParams.restartUpdateRuleInTransfer = adi;
hyperParams.transferSoftmax = true;
hyperParams.parensInSequences = 0;
hyperParams.dataPortion = dp;
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;

if wordsource == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif wordsource == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.'); 
elseif wordsource == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
end

hyperParams.useEmbeddingTransform = true;
hyperParams.topDepth = topDepth;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

if gc > 0
    hyperParams.clipGradients = true;
    hyperParams.maxGradNorm = gc;
end

% Apply dropout to the top feature vector of each tree, preserving activations
% with this probability. If this is set to 1, dropout is effectively not used.
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;
hyperParams = CompositionSetup(hyperParams, composition);
hyperParams.useThirdOrderMerge = false;

% Small corpus, report often.
options.costFreq = 100;
options.testFreq = 100;
options.detailedStatFreq = 100;

hyperParams.numLabels = [2];

hyperParams.labels = {{'True', 'False'}};
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

hyperParams.trainFilenames = {'../data/rte3_train_parsed.tab'};    
hyperParams.splitFilenames = {};    
hyperParams.testFilenames = {'../data/rte3_test_parsed.tab'};

wordMap = LoadWordMap('../data/rte3_words.txt');
hyperParams.vocabName = 'rte3'; 

if strcmp(dataflag, 'rte3')
    % Tranfer wordlist is same.
elseif strcmp(dataflag, 'rte3-transfer')
    hyperParams.sourceWordMap = LoadWordMap('../data/snlirc3_words.txt');
end

end
