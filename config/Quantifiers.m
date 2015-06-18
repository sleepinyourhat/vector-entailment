function [ hyperParams, options, wordMap, relationMap ] = Quantifiers(name, dataflag, dim, penult, top, lambda, tot, relu, tdrop, mbs)

[hyperParams, options] = Defaults();

hyperParams.name = [name, '-', dataflag, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-tot', num2str(tot), '-relu', num2str(relu), '-l', num2str(lambda),...
				    '-dropout', num2str(tdrop), '-mb', num2str(mbs)];

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% Use NTN layers in place of NN layers.
if tot > 0
	hyperParams.useThirdOrder = tot;
	hyperParams.useThirdOrderComparison = tot;
else
	hyperParams.useSumming = true;
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 0;
end

hyperParams.topDepth = top;

hyperParams.topDropout = tdrop;

if relu
  hyperParams.classNL = @LReLU;
  hyperParams.classNLDeriv = @LReLUDeriv;
end

wordMap = InitializeMaps('./grammars/wordlist.tsv'); 
hyperParams.vocabName = 'quantifiers'

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

if strcmp(dataflag, 'f1')
	hyperParams.trainFilenames = {'./grammars/data/f1_train.txt'};
	hyperParams.testFilenames = {'./grammars/data/f1_test.txt'};
elseif strcmp(dataflag, 'f2')
	hyperParams.trainFilenames = {'./grammars/data/f2_train.txt'};
	hyperParams.testFilenames = {'./grammars/data/f2_test.txt'};
elseif strcmp(dataflag, 'f3')
	hyperParams.trainFilenames = {'./grammars/data/f3_train.txt'};
	hyperParams.testFilenames = {'./grammars/data/f3_test.txt'};
elseif strcmp(dataflag, 'f4')
	hyperParams.trainFilenames = {'./grammars/data/f4_train.txt'};
	hyperParams.testFilenames = {'./grammars/data/f4_test.txt'};
elseif strcmp(dataflag, 'f5')
	hyperParams.trainFilenames = {'./grammars/data/f5_train.txt'};
	hyperParams.testFilenames = {'./grammars/data/f5_test.txt'};
else
	hyperParams.trainFilenames = {'./grammars/data/dev_train.txt'};
	hyperParams.testFilenames = {'./grammars/data/dev_test.txt'};
end
hyperParams.splitFilenames = {};

options.numPasses = 1000;
options.examplesFreq = 25000; 
options.numPasses = 2500;

options.miniBatchSize = mbs;

end
