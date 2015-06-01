function [ hyperParams, options, wordMap, labelMap ] = SynsetRelations(name, transDepth, penult, lambda, tot, mbs, lr, trainwords, fastemb)
% Configuration for WordNet experiments.
% See Defaults.m for parameter descriptions.

[hyperParams, options] = Defaults();

hyperParams.name = name;

hyperParams.dim = 25;
hyperParams.useEmbeddingTransform = transDepth;
hyperParams.penultDim = penult;
hyperParams.lambda = lambda;
hyperParams.useThirdOrderComposition = tot;
hyperParams.useThirdOrderMerge = tot;
hyperParams.loadWords = true;
hyperParams.trainWords = trainwords;
hyperParams.largeVocabMode = fastemb;

options.miniBatchSize = mbs;
options.lr = lr;

hyperParams.numLabels = [3]; 

wordMap = LoadWordMap('synset-labels/longer_wordlist.txt');
hyperParams.vocabName = 'synset'; 

hyperParams.labels = {{'hypernym', 'hyponym', 'coordinate'}};
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

hyperParams.trainFilenames = {};
hyperParams.testFilenames = {};
hyperParams.splitFilenames = {'./synset-labels/longer_shuffled_synset_labels.tsv'};

end
