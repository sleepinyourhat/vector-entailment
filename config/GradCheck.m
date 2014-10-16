function [ hyperParams, options, wordMap, relationMap ] = AndOr(transDepth, topDepth, tot, trainwords)

[hyperParams, options] = Defaults();

% The dimensionality of the word/phrase vectors. Currently fixed at 25 to match
% the GloVe vectors.
hyperParams.dim = 2;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = 2;

% The number of embedding transform layers. transDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = transDepth;

hyperParams.topDepth = topDepth;

% Regularization coefficient.
hyperParams.lambda = 0.02;

% Use NTN layers in place of NN layers.
hyperParams.useThirdOrder = tot;
hyperParams.useThirdOrderComparison = tot;

hyperParams.loadWords = false;
hyperParams.trainWords = true;

hyperParams.minFunc = true;

%%% minFunc options:
global options
options.Method = 'lbfgs';
options.MaxFunEvals = 1000;
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options.DerivativeCheck = 'on';
% options.OutputFcn = @Display;  % Custom error reporting for minFunc

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = 1;

options.numPasses = 1;

options.lr = 0.1;

wordMap = InitializeMaps('./grammars/wordlist.tsv'); 
hyperParams.vocabName = 'quantifiers'

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

hyperParams.splitFilenames = {'./grammars/test_file.tsv'};
hyperParams.trainFilenames = {};
hyperParams.testFilenames = {};

end