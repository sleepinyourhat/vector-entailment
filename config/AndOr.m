function [ hyperParams, options, wordMap, relationMap ] = AndOr(name, dataflag, dim, penult, top, lambda, tot, relu, tdrop, mbs)
% Configure the recursion experiments. 
% NOTE: the {a-h} variables in the paper are actual multiletter names in the data used here.

[hyperParams, options] = Defaults();


hyperParams.name = [name, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-tot', num2str(tot), '-relu', num2str(relu), '-l', num2str(lambda), ...
				    '-dropout', num2str(tdrop), '-mb', num2str(mbs)];

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;


% The raw range bound on word vectors.
hyperParams.wordScale = 0.01;

% Used to compute the bound on the range for RNTN parameter initialization.
hyperParams.tensorScale = 1;

% Use an older initialization scheme for comparability with older experiments.
hyperParams.useCompatibilityInitialization = true;

hyperParams.useEyes = 1;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda;


if tot == -1
  hyperParams.useThirdOrder = 0;
  hyperParams.useThirdOrderComparison = 0;
  hyperParams.useSumming = 1;
elseif tot < 2
	hyperParams.useThirdOrder = tot;
	hyperParams.useThirdOrderComparison = tot;
elseif tot == 2
	hyperParams.lstm = 1;
	hyperParams.useTrees = 0;
	hyperParams.eyeScale = 0;
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 0;
	hyperParams.parensInSequences = 1;
elseif tot == 3
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 0;
	hyperParams.parensInSequences = 1;
end

% Add identity matrices where appropriate in initiazilation.
hyperParams.eyeScale = hyperParams.eyeScale * (1 - hyperParams.lstm);

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

wordMap = InitializeMaps('/scr/sbowman/RC/longer2/wordlist.txt');

% The name assigned to the current vocabulary. Used in deciding whether to load a 
% preparsed MAT form of an examples file.
hyperParams.vocabName = 'RC'; 

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = 7;
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

if strcmp(dataflag, 'and-or') 
    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'/scr/sbowman/RC/train0', '/scr/sbowman/RC/train1', '/scr/sbowman/RC/train2', '/scr/sbowman/RC/train3', '/scr/sbowman/RC/train4', '/scr/sbowman/RC/test0', '/scr/sbowman/RC/test1', '/scr/sbowman/RC/test2', '/scr/sbowman/RC/test3', '/scr/sbowman/RC/test4', '/scr/sbowman/RC/test5', '/scr/sbowman/RC/test6'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep') 
    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'/scr/sbowman/RC/longer2/train0', '/scr/sbowman/RC/longer2/train1', '/scr/sbowman/RC/longer2/train2', '/scr/sbowman/RC/longer2/train3', '/scr/sbowman/RC/longer2/train4', '/scr/sbowman/RC/longer2/test1', '/scr/sbowman/RC/longer2/test2', '/scr/sbowman/RC/longer2/test3', '/scr/sbowman/RC/longer2/test4', '/scr/sbowman/RC/longer2/test5', '/scr/sbowman/RC/longer2/test6', '/scr/sbowman/RC/longer2/test7', '/scr/sbowman/RC/longer2/test8', '/scr/sbowman/RC/longer2/test9', '/scr/sbowman/RC/longer2/test10', '/scr/sbowman/RC/longer2/test11', '/scr/sbowman/RC/longer2/test12'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep-static') 
    hyperParams.trainFilenames = {'/scr/sbowman/RC/longer2/train0', '/scr/sbowman/RC/longer2/train1', '/scr/sbowman/RC/longer2/train2', '/scr/sbowman/RC/longer2/train3', '/scr/sbowman/RC/longer2/train4'};
    hyperParams.splitFilenames = {};
    hyperParams.testFilenames = {'/scr/sbowman/RC/longer2/test1', '/scr/sbowman/RC/longer2/test2', '/scr/sbowman/RC/longer2/test3', '/scr/sbowman/RC/longer2/test4', '/scr/sbowman/RC/longer2/test5', '/scr/sbowman/RC/longer2/test6', '/scr/sbowman/RC/longer2/test7', '/scr/sbowman/RC/longer2/test8', '/scr/sbowman/RC/longer2/test9', '/scr/sbowman/RC/longer2/test10', '/scr/sbowman/RC/longer2/test11', '/scr/sbowman/RC/longer2/test12'};
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
options.examplesFreq = 25000; 


end
