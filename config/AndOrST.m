function [ hyperParams, options, wordMap, relationMap ] = AndOrST(name, dataflag, dim, penult, top, lambda, tot, relu, tdrop, mbs, lstm)
% Configure the recursion experiments. 
% NOTE: the {a-h} variables in the paper are actual multiletter names in the data used here.

[hyperParams, options] = Defaults();

hyperParams.useTrees = 0;
hyperParams.parensInSequences = 1;
hyperParams.lstm = lstm;
hyperParams.useEyes = 0;

if length(name) == 0
	name = 'and-or'
end

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
hyperParams.useThirdOrder = 0;
hyperParams.useThirdOrderComparison = 0;


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

wordMap = InitializeMaps('./RC/longer2/wordlist.txt');

% The name assigned to the current vocabulary. Used in deciding whether to load a 
% preparsed MAT form of an examples file.
hyperParams.vocabName = 'RC'; 

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = 7;
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

hyperParams.trainFilenames = {};
hyperParams.splitFilenames = {'./RC/test0', './RC/test1', './RC/test2'};
hyperParams.testFilenames = {};

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
