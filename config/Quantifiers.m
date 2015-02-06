function [ hyperParams, options, wordMap, relationMap ] = Quantifiers(name, dim, penult, top, lambda, tot, eyes, tdrop, mbs)

[hyperParams, options] = Defaults();

hyperParams.name = [name, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-tot', num2str(tot), '-eyes', num2str(eyes), '-l', num2str(lambda),...
				    '-dropout', num2str(tdrop), '-mb', num2str(mbs)];

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works for Tree, 1e-6 for Sequence?

hyperParams.eyeScale = eyes;

% Use NTN layers in place of NN layers.
if tot < 2
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