function [ hyperParams, options, wordMap, relationMap ] = Sick(dataflag, penult, lambda, tot, mbs, lr)

[hyperParams, options] = Defaults();

% the GloVe vectors.
hyperParams.dim = 25;

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = 0;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% Use NTN layers in place of NN layers.
hyperParams.useThirdOrder = tot;
hyperParams.useThirdOrderComparison = tot;

hyperParams.loadWords = false;
hyperParams.trainWords = true;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = mbs;

options.lr = lr;

wordMap = InitializeMaps('./grammars/wordlist.tsv'); 
hyperParams.vocabName = 'quantifiers'

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

listingG = dir('grammars/data/quant*');
hyperParams.splitFilenames = {};
hyperParams.trainFilenames = {};
hyperParams.testFilenames = {};

elseif strcmp(dataflag, 'G-two_lt_two')
    hyperParams.testFilenames = {'grammars/data/quant_two_lt_two'};
    hyperParams.splitFilenames = {listingG.name};
elseif strcmp(dataflag, 'G-no_no')
    hyperParams.testFilenames = {'grammars/data/quant_no_no'};
    hyperParams.splitFilenames = {listingG.name};
elseif strcmp(dataflag, 'G-not_all_not_most')
    hyperParams.testFilenames = {'grammars/data/quant_not_all_not_most'};
    hyperParams.splitFilenames = {listingG.name};
elseif strcmp(dataflag, 'G-all_some')
    hyperParams.testFilenames = {'grammars/data/quant_all_some'};
    hyperParams.splitFilenames = {listingG.name};
elseif strcmp(dataflag, 'G-splitall')
    hyperParams.splitFilenames = {listingG.name};
end

end