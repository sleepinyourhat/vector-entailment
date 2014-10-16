function [ hyperParams, options, wordMap, relationMap ] = Sick(dataflag, transDepth, penult, lambda, tot, mbs, lr, trainwords)

[hyperParams, options] = Defaults();

% The dimensionality of the word/phrase vectors. Currently fixed at 25 to match
% the GloVe vectors.
hyperParams.dim = 25;

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = transDepth;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% Use NTN layers in place of NN layers.
hyperParams.useThirdOrder = tot;
hyperParams.useThirdOrderComparison = tot;

hyperParams.loadWords = true;
hyperParams.trainWords = trainwords;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = mbs;

options.lr = lr;

hyperParams.numRelations = [3]; 

wordMap = InitializeMaps('synset-relations/longer_wordlist.txt');
hyperParams.vocabName = 'synset'; 

hyperParams.relations = {{'hypernym', 'hyponym', 'coordinate'}};
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

hyperParams.trainFilenames = {};
hyperParams.testFilenames = {};
hyperParams.splitFilenames = {'./synset-relations/longer_shuffled_synset_relations.tsv'};

end