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
  hyperParams.useThirdOrderComposition = 0;
  hyperParams.useThirdOrderMerge = 0;
  hyperParams.useSumming = 1;
elseif composition < 2
  hyperParams.useThirdOrderComposition = composition;
  hyperParams.useThirdOrderMerge = composition;
elseif composition == 2
  hyperParams.lstm = 1;
  hyperParams.useTrees = 0;
  hyperParams.eyeScale = 0;
  hyperParams.useThirdOrderComposition = 0;
  hyperParams.useThirdOrderMerge = 1;
  hyperParams.parensInSequences = 1;
elseif composition == 3
  hyperParams.lstm = 0;
  hyperParams.useTrees = 0;
  hyperParams.useThirdOrderComposition = 0;
  hyperParams.useThirdOrderMerge = 1;
  hyperParams.parensInSequences = 1;
elseif composition == 4
  hyperParams.useLattices = 1;
  hyperParams.lstm = 0;
  hyperParams.useTrees = 0;
  hyperParams.useThirdOrderComposition = 0;
  hyperParams.useThirdOrderMerge = 0;
elseif composition == 5
  hyperParams.useLattices = 1;
  hyperParams.lstm = 0;
  hyperParams.useTrees = 0;
  hyperParams.useThirdOrderComposition = 0;
  hyperParams.useThirdOrderMerge = 1;
end


hyperParams.topDropout = 1;

hyperParams.topDepth = top;

options.numPasses = 15000;

options.miniBatchSize = mbs;

wordMap = InitializeMaps('./propositionallogic/longer2/wordlist.txt');

% The name assigned to the current vocabulary. Used in deciding whether to load a 
% preparsed MAT form of an examples file.
hyperParams.vocabName = 'RC'; 

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

hyperParams.ignorePreprocessedFiles = true;

if strcmp(dataflag, 'and-or') 
    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'./propositionallogic/train0', './propositionallogic/train1', './propositionallogic/train2', './propositionallogic/train3', './propositionallogic/train4', './propositionallogic/test0', './propositionallogic/test1', './propositionallogic/test2', './propositionallogic/test3', './propositionallogic/test4', './propositionallogic/test5', './propositionallogic/test6'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep') 
	% Split longer test sets for crossvalidation without training on them.
	hyperParams.specialAndOrMode = 4;

    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'./propositionallogic/longer2/train0', './propositionallogic/longer2/train1', './propositionallogic/longer2/train2', './propositionallogic/longer2/train3', './propositionallogic/longer2/train4', './propositionallogic/longer2/test1', './propositionallogic/longer2/test2', './propositionallogic/longer2/test3', './propositionallogic/longer2/test4', './propositionallogic/longer2/test5', './propositionallogic/longer2/test6', './propositionallogic/longer2/test7', './propositionallogic/longer2/test8', './propositionallogic/longer2/test9', './propositionallogic/longer2/test10', './propositionallogic/longer2/test11', './propositionallogic/longer2/test12'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep-3') 
	% Split longer test sets for crossvalidation without training on them.
	hyperParams.specialAndOrMode = 3;

    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'./propositionallogic/longer2/train0', './propositionallogic/longer2/train1', './propositionallogic/longer2/train2', './propositionallogic/longer2/train3', './propositionallogic/longer2/test1', './propositionallogic/longer2/test2', './propositionallogic/longer2/test3', './propositionallogic/longer2/test4', './propositionallogic/longer2/test5', './propositionallogic/longer2/test6', './propositionallogic/longer2/test7', './propositionallogic/longer2/test8', './propositionallogic/longer2/test9', './propositionallogic/longer2/test10', './propositionallogic/longer2/test11', './propositionallogic/longer2/test12'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep-6') 
	% Split longer test sets for crossvalidation without training on them.
	hyperParams.specialAndOrMode = 6;

    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'./propositionallogic/longer2/train0', './propositionallogic/longer2/train1', './propositionallogic/longer2/train2', './propositionallogic/longer2/train3', './propositionallogic/longer2/train4', './propositionallogic/longer2/train5', './propositionallogic/longer2/train6', './propositionallogic/longer2/test1', './propositionallogic/longer2/test2', './propositionallogic/longer2/test3', './propositionallogic/longer2/test4', './propositionallogic/longer2/test5', './propositionallogic/longer2/test6', './propositionallogic/longer2/test7', './propositionallogic/longer2/test8', './propositionallogic/longer2/test9', './propositionallogic/longer2/test10', './propositionallogic/longer2/test11', './propositionallogic/longer2/test12'};
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'and-or-deep-static') 
    hyperParams.trainFilenames = {'./propositionallogic/longer2/train0', './propositionallogic/longer2/train1', './propositionallogic/longer2/train2', './propositionallogic/longer2/train3', './propositionallogic/longer2/train4'};
    hyperParams.splitFilenames = {};
    hyperParams.testFilenames = {'./propositionallogic/longer2/test1', './propositionallogic/longer2/test2', './propositionallogic/longer2/test3', './propositionallogic/longer2/test4', './propositionallogic/longer2/test5', './propositionallogic/longer2/test6', './propositionallogic/longer2/test7', './propositionallogic/longer2/test8', './propositionallogic/longer2/test9', './propositionallogic/longer2/test10', './propositionallogic/longer2/test11', './propositionallogic/longer2/test12'};
end

options.detailedStatFreq = 1000;
options.costFreq = 1000;
options.testFreq = 1000;
options.examplesFreq = 25000; 

end
