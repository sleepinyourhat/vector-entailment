function [ hyperParams, options, wordMap, relationMap ] = AndOr(name, dataflag, dim, penult, top, lambda, tot, relu, tdrop, mbs)
% Configure the recursion experiments. 
% NOTE: the {a-h} variables in the paper are actual multiletter names in the data used here.

[hyperParams, options] = Defaults();

hyperParams.name = [name, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-tot', num2str(tot), '-relu', num2str(relu), '-l', num2str(lambda), ...
				    '-dropout', num2str(tdrop), '-mb', num2str(mbs)];

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda;

% Use NTN layers in place of NN layers.
if tot == -1
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 0;
	hyperParams.useSumming = 1
elseif tot == 0
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 0;
elseif tot == 1
	hyperParams.useThirdOrder = 1;
	hyperParams.useThirdOrderComparison = 1;
elseif tot == 2
	hyperParams.useThirdOrder = 1;
	hyperParams.useThirdOrderComparison = 0;
elseif tot == 3
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 1;
end

hyperParams.topDropout = tdrop;

hyperParams.topDepth = top;

% Split longer test sets for crossvalidation without training on them.
hyperParams.specialAndOrMode = 1;

if relu
  hyperParams.classNL = @LReLU;
  hyperParams.classNLDeriv = @LReLUDeriv;
end

options.numPasses = 15000;

options.miniBatchSize = mbs;

wordMap = LoadTrainingData('./RC/longer2/train1');

% The name assigned to the current vocabulary. Used in deciding whether to load a 
% preparsed MAT form of an examples file.
hyperParams.vocabName = 'RC'; 

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = 7;
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

% TODO: Set up folds

if strcmp(dataflag, 'and-or') 
    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'./RC/longer2/train0', './RC/longer2/train1', './RC/longer2/train2', './RC/longer2/train3', './RC/longer2/train4', './RC/longer2/train5'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep') 
    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'./RC/longer2/train0', './RC/longer2/train1', './RC/longer2/train2', './RC/longer2/train3', './RC/longer2/train4', './RC/longer2/test1', './RC/longer2/test2', './RC/longer2/test3', './RC/longer2/test4', './RC/longer2/test5', './RC/longer2/test6', './RC/longer2/test7', './RC/longer2/test8', './RC/longer2/test9', './RC/longer2/test10', './RC/longer2/test11', './RC/longer2/test12'};
    hyperParams.testFilenames = {};
end


% How often (in steps) to report cost.
options.costFreq = 1000;

% How often (in steps) to run on test data.
options.testFreq = 1000;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 1000;

% How often to display which items are misclassified.
% Should be a multiple of testFreq.
options.examplesFreq = 10000; 


end
