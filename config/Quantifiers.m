function [ hyperParams, options, wordMap, relationMap ] = Quantifiers(name, dim, penult, top, lambda, tot, eyes, tdrop, mbs)

[hyperParams, options] = Defaults();

hyperParams.name = [name, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-tot', num2str(tot), '-eyes', num2str(eyes), '-l', num2str(lambda),...
				    '-dropout', num2str(tdrop), '-mb', num2str(mbs)];

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

% The raw range bound on word vectors.
hyperParams.wordScale = 0.01;

% Used to compute the bound on the range for RNTN parameter initialization.
hyperParams.tensorScale = 1;

% Use an older initialization scheme for comparability with older experiments.
hyperParams.useCompatibilityInitialization = true;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works for Tree, 1e-6 for Sequence?

hyperParams.useEyes = 1;
hyperParams.eyeScale = eyes;

% Use NTN layers in place of NN layers.
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
	hyperParams.useThirdOrderComparison = 1;
	hyperParams.parensInSequences = 0;
elseif tot == 3
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrder = 0;
	hyperParams.useThirdOrderComparison = 1;
	hyperParams.parensInSequences = 0;
end
	

hyperParams.topDepth = top;

hyperParams.topDropout = tdrop;

hyperParams.classNL = @LReLU;
hyperParams.classNLDeriv = @LReLUDeriv;

wordMap = InitializeMaps('./grammars/wordlist.tsv'); 
hyperParams.vocabName = 'quantifiers'

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

listingG = dir('grammars/data/quant_*');
hyperParams.trainFilenames = {};
hyperParams.testFilenames = {};
hyperParams.splitFilenames = {listingG.name};

options.numPasses = 1000;

options.miniBatchSize = mbs;

end