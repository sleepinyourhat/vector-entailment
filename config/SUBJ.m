function [ hyperParams, options, wordMap, labelMap ] = SUBJ(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, wordsource, adi)
% Configuration for subjectivity classification experiments.
% See Defaults.m for parameter descriptions.

[hyperParams, options] = Defaults();
hyperParams.sentenceClassificationMode = true;
hyperParams.largeVocabMode = true;
hyperParams.loadWords = true;
hyperParams.trainWords = true;

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(wordsource),...
    '-comp', num2str(composition), '-adi', num2str(adi)  ];

hyperParams.restartUpdateRuleInTransfer = adi;
hyperParams.testFraction = 0.1;
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;
hyperParams.useEmbeddingTransform = true;
hyperParams.topDepth = topDepth;
hyperParams.penultDim = penult;
hyperParams.lambda = lambda;
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;
hyperParams = CompositionSetup(hyperParams, composition);

if wordsource == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif wordsource == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.'); 
elseif wordsource == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
end

options.miniBatchSize = 32;

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
	hyperParams.sourceWordMap = LoadWordMap('../data/snlirc3_words.txt');
end
end
