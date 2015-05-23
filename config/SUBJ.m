function [ hyperParams, options, wordMap, labelMap ] = SUBJ(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, collo, adi)

[hyperParams, options] = Defaults();

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(collo),...
    '-comp', num2str(composition), '-adi', num2str(adi)  ];

hyperParams.restartUpdateRuleInTransfer = adi;
%TEMP:
options.updateFn = @RMSPropUpdate;


hyperParams.sentenceClassificationMode = 1;

hyperParams.testFraction = 0.1;

hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;

if collo == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif collo == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.'); 
elseif collo == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
end

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = 1;

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = topDepth;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step.
hyperParams.largeVocabMode = true;

% The dimensionality of the classifier extra layers.
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = 32;

% Apply dropout to the top feature vector of each tree, preserving activations
% with this probability. If this is set to 1, dropout is effectively not used.
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;

hyperParams = CompositionSetup(hyperParams, composition);

hyperParams.loadWords = true;
hyperParams.trainWords = true;

hyperParams.fragmentData = false;

wordMap = LoadWordMap('../data/subj_words.txt');
hyperParams.vocabName = 'subj'; 

hyperParams.numLabels = [ 2 ];

hyperParams.labels = {{'subjective', 'objective'}};
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

hyperParams.trainFilenames = {};    
hyperParams.splitFilenames = {'../data/subj_parsed.txt'};    
hyperParams.testFilenames = {};


if strcmp(dataflag, 'subj')
elseif strcmp(dataflag, 'subj-transfer')
	hyperParams.sourceWordMap = LoadWordMap('../data/snlirc2_words.txt');
end
end
