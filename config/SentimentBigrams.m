function [ hyperParams, options, wordMap, labelMap ] = SentimentBigrams(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, wordsource, adi)
% Configuration for bigram sentiment classification experiments.
% See Defaults.m for parameter descriptions.

% Note: Use only summing-based composition setups.

[hyperParams, options] = Defaults();
hyperParams.sentenceClassificationMode = false;  % Treat word pairs as sentence pairs.
hyperParams.sentimentBigramMode = true;
hyperParams.largeVocabMode = true;
hyperParams.loadWords = false;
hyperParams.trainWords = true;

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(wordsource),...
    '-comp', num2str(composition)];

hyperParams.testFraction = 0.2;
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;
hyperParams.useEmbeddingTransform = false;
hyperParams.topDepth = topDepth;
hyperParams.penultDim = penult;
hyperParams.lambda = lambda;
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;
hyperParams = CompositionSetup(hyperParams, composition);

options.miniBatchSize = 32;
options.detailedStatFreq = 5000;

wordMap = LoadWordMap('../data/sentiment-bigrams_words.txt');
hyperParams.vocabName = 'sentb-'; 

hyperParams.numLabels = [ 10 ];

hyperParams.labels = {{'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}};
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

hyperParams.trainFilenames = {'../data/imdb_bigrams_train.txt'};    
hyperParams.splitFilenames = {};    
hyperParams.testFilenames = {'../data/imdb_bigrams_dev.txt', '../data/imdb_bigrams_test.txt'};

end
