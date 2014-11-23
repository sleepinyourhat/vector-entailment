function [ hyperParams, options, wordMap, relationMap ] = AndOr(name, dataflag, transDepth, penult, lambda, tot, mbs, lr, trainwords)

% TODO: Assign wordMap.
[hyperParams, options] = Defaults();

% The dimensionality of the word/phrase vectors. Currently fixed at 25 to match
% the GloVe vectors.
hyperParams.dim = dim;

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

options.numPasses = 15000;

options.lr = lr;

wordMap = LoadTrainingData('./RC/train1');
% The name assigned to the current vocabulary. Used in deciding whether to load a 
% preparsed MAT form of an examples file.
hyperParams.vocabName = 'RC'; 

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

if strcmp(dataflag, 'and-or') 
    hyperParams.trainFilenames = {'./RC/train0', './RC/train1', './RC/train2', './RC/train3', './RC/train4'};
    hyperParams.testFilenames = {'./RC/test0', './RC/test1', './RC/test2', './RC/test3', './RC/test4', './RC/test5', './RC/test6'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'and-or-deep') 
    hyperParams.trainFilenames = {'./RC/longer2/train0', './RC/longer2/train1', './RC/longer2/train2', './RC/longer2/train3', './RC/longer2/train4'};
    hyperParams.testFilenames = {'./RC/longer2/test0', './RC/longer2/test1', './RC/longer2/test2', './RC/longer2/test3', './RC/longer2/test4', './RC/longer2/test5', './RC/longer2/test6', './RC/longer2/test7', './RC/longer2/test8', './RC/longer2/test9', './RC/longer2/test10', './RC/longer2/test11', './RC/longer2/test12'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'and-or-deep-unlim') 
    hyperParams.trainFilenames = {'./RC/longer2/train0', './RC/longer2/train1', './RC/longer2/train2', './RC/longer2/train3', './RC/longer2/train4', './RC/longer2/train5', './RC/longer2/train6', './RC/longer2/train7', './RC/longer2/train8', './RC/longer2/train9', './RC/longer2/train10', './RC/longer2/train11', './RC/longer2/train12'}; 
    hyperParams.testFilenames = {'./RC/longer2/test0', './RC/longer2/test1', './RC/longer2/test2', './RC/longer2/test3', './RC/longer2/test4', './RC/longer2/test5', './RC/longer2/test6', './RC/longer2/test7', './RC/longer2/test8', './RC/longer2/test9', './RC/longer2/test10', './RC/longer2/test11', './RC/longer2/test12'};
    hyperParams.splitFilenames = {};
end

end