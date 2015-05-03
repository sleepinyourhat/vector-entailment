function [ hyperParams, options, wordMap, labelMap ] = SynsetLabels(name, transDepth, penult, lambda, tot, mbs, lr, trainwords, fastemb)

[hyperParams, options] = Defaults();

hyperParams.name = name;

% The dimensionality of the word/phrase vectors.
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
hyperParams.useThirdOrderComposition = tot;
hyperParams.useThirdOrderMerge = tot;

hyperParams.loadWords = true;
hyperParams.trainWords = trainwords;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step.
hyperParams.fastEmbed = fastemb;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = mbs;

options.lr = lr;

hyperParams.numLabels = [3]; 

wordMap = InitializeMaps('synset-labels/longer_wordlist.txt');
hyperParams.vocabName = 'synset'; 

hyperParams.labels = {{'hypernym', 'hyponym', 'coordinate'}};
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

hyperParams.trainFilenames = {};
hyperParams.testFilenames = {};
hyperParams.splitFilenames = {'./synset-labels/longer_shuffled_synset_labels.tsv'};

end
