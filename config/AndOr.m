function [ hyperParams, options, wordMap, relationMap ] = AndOr(name, dataflag, dim, penult, top, lambda, composition, mbs)
% Configure the recursion experiments. 
% NOTE: the {a-h} variables in the paper are actual multiletter names in the data used here.

[hyperParams, options] = Defaults();

hyperParams.name = [name, '-', dataflag, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-comp', num2str(composition), '-mbs', num2str(mbs), '-l', num2str(lambda)];

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda;

if composition == -1
  hyperParams.useTrees = 0;
  hyperParams.useThirdOrder = 0;
  hyperParams.useThirdOrderComparison = 0;
  hyperParams.useSumming = 1;
elseif composition < 2
	hyperParams.useThirdOrder = composition;
	hyperParams.useThirdOrderComparison = composition;
elseif composition == 2
	hyperParams.lstm = 1;
	hyperParams.useTrees = 0;
	hyperParams.eyeScale = 0;
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 0;
	hyperParams.parensInSequences = 1;
elseif composition == 3
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 0;
	hyperParams.parensInSequences = 1;
end

hyperParams.topDropout = 1;

hyperParams.topDepth = top;

options.numPasses = 15000;

options.miniBatchSize = mbs;

wordMap = InitializeMaps('/scr/sbowman/propositional-logic/longer2/wordlist.txt');

% The name assigned to the current vocabulary. Used in deciding whether to load a 
% preparsed MAT form of an examples file.
hyperParams.vocabName = 'RC'; 

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

if strcmp(dataflag, 'and-or') 
    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'/scr/sbowman/propositional-logic/train0', '/scr/sbowman/propositional-logic/train1', '/scr/sbowman/propositional-logic/train2', '/scr/sbowman/propositional-logic/train3', '/scr/sbowman/propositional-logic/train4', '/scr/sbowman/propositional-logic/test0', '/scr/sbowman/propositional-logic/test1', '/scr/sbowman/propositional-logic/test2', '/scr/sbowman/propositional-logic/test3', '/scr/sbowman/propositional-logic/test4', '/scr/sbowman/propositional-logic/test5', '/scr/sbowman/propositional-logic/test6'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep') 
	% Split longer test sets for crossvalidation without training on them.
	hyperParams.specialAndOrMode = 4;

    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'/scr/sbowman/propositional-logic/longer2/train0', '/scr/sbowman/propositional-logic/longer2/train1', '/scr/sbowman/propositional-logic/longer2/train2', '/scr/sbowman/propositional-logic/longer2/train3', '/scr/sbowman/propositional-logic/longer2/train4', '/scr/sbowman/propositional-logic/longer2/test1', '/scr/sbowman/propositional-logic/longer2/test2', '/scr/sbowman/propositional-logic/longer2/test3', '/scr/sbowman/propositional-logic/longer2/test4', '/scr/sbowman/propositional-logic/longer2/test5', '/scr/sbowman/propositional-logic/longer2/test6', '/scr/sbowman/propositional-logic/longer2/test7', '/scr/sbowman/propositional-logic/longer2/test8', '/scr/sbowman/propositional-logic/longer2/test9', '/scr/sbowman/propositional-logic/longer2/test10', '/scr/sbowman/propositional-logic/longer2/test11', '/scr/sbowman/propositional-logic/longer2/test12'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep-3') 
	% Split longer test sets for crossvalidation without training on them.
	hyperParams.specialAndOrMode = 3;

    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'/scr/sbowman/propositional-logic/longer2/train0', '/scr/sbowman/propositional-logic/longer2/train1', '/scr/sbowman/propositional-logic/longer2/train2', '/scr/sbowman/propositional-logic/longer2/train3', '/scr/sbowman/propositional-logic/longer2/test1', '/scr/sbowman/propositional-logic/longer2/test2', '/scr/sbowman/propositional-logic/longer2/test3', '/scr/sbowman/propositional-logic/longer2/test4', '/scr/sbowman/propositional-logic/longer2/test5', '/scr/sbowman/propositional-logic/longer2/test6', '/scr/sbowman/propositional-logic/longer2/test7', '/scr/sbowman/propositional-logic/longer2/test8', '/scr/sbowman/propositional-logic/longer2/test9', '/scr/sbowman/propositional-logic/longer2/test10', '/scr/sbowman/propositional-logic/longer2/test11', '/scr/sbowman/propositional-logic/longer2/test12'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep-6') 
	% Split longer test sets for crossvalidation without training on them.
	hyperParams.specialAndOrMode = 6;

    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'/scr/sbowman/propositional-logic/longer2/train0', '/scr/sbowman/propositional-logic/longer2/train1', '/scr/sbowman/propositional-logic/longer2/train2', '/scr/sbowman/propositional-logic/longer2/train3', '/scr/sbowman/propositional-logic/longer2/train4', '/scr/sbowman/propositional-logic/longer2/train5', '/scr/sbowman/propositional-logic/longer2/train6', '/scr/sbowman/propositional-logic/longer2/test1', '/scr/sbowman/propositional-logic/longer2/test2', '/scr/sbowman/propositional-logic/longer2/test3', '/scr/sbowman/propositional-logic/longer2/test4', '/scr/sbowman/propositional-logic/longer2/test5', '/scr/sbowman/propositional-logic/longer2/test6', '/scr/sbowman/propositional-logic/longer2/test7', '/scr/sbowman/propositional-logic/longer2/test8', '/scr/sbowman/propositional-logic/longer2/test9', '/scr/sbowman/propositional-logic/longer2/test10', '/scr/sbowman/propositional-logic/longer2/test11', '/scr/sbowman/propositional-logic/longer2/test12'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep-static') 
    hyperParams.trainFilenames = {'/scr/sbowman/propositional-logic/longer2/train0', '/scr/sbowman/propositional-logic/longer2/train1', '/scr/sbowman/propositional-logic/longer2/train2', '/scr/sbowman/propositional-logic/longer2/train3', '/scr/sbowman/propositional-logic/longer2/train4'};
    hyperParams.splitFilenames = {};
    hyperParams.testFilenames = {'/scr/sbowman/propositional-logic/longer2/test1', '/scr/sbowman/propositional-logic/longer2/test2', '/scr/sbowman/propositional-logic/longer2/test3', '/scr/sbowman/propositional-logic/longer2/test4', '/scr/sbowman/propositional-logic/longer2/test5', '/scr/sbowman/propositional-logic/longer2/test6', '/scr/sbowman/propositional-logic/longer2/test7', '/scr/sbowman/propositional-logic/longer2/test8', '/scr/sbowman/propositional-logic/longer2/test9', '/scr/sbowman/propositional-logic/longer2/test10', '/scr/sbowman/propositional-logic/longer2/test11', '/scr/sbowman/propositional-logic/longer2/test12'};
end


% How often (in steps) to report cost.
options.detailedStatFreq = 1000;

% How often (in steps) to run on test data.
options.testFreq = 1000;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 1000;

% How often to display which items are misclassified.
% Should be a multiple of testFreq.
options.examplesFreq = 25000; 


end
