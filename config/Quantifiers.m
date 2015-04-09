function [ hyperParams, options, wordMap, relationMap ] = Quantifiers(name, dim, penult, top, lambda, composition, eyes, tdrop)

[hyperParams, options] = Defaults();

hyperParams.name = [name, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-composition', num2str(composition), '-eyes', num2str(eyes), '-l', num2str(lambda),...
				    '-dropout', num2str(tdrop)];

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works for Tree, 1e-6 for Sequence?

if composition == -1
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
	hyperParams.parensInSequences = 0;
elseif composition == 3
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 1;
	hyperParams.parensInSequences = 0;
elseif composition == 4
	hyperParams.usePyramids = 1;
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 0;
	hyperParams.parensInSequences = 0;
elseif composition == 5
	hyperParams.usePyramids = 1;
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 1;
	hyperParams.parensInSequences = 0;
end


hyperParams.topDepth = top;

hyperParams.topDropout = tdrop;

wordMap = InitializeMaps('./quantifiers/wordlist.tsv'); 
hyperParams.vocabName = 'quantifiers'

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

listingG = dir('./quantifiers/data/quant_*');
hyperParams.trainFilenames = {};
hyperParams.testFilenames = {};
hyperParams.splitFilenames = strcat('./quantifiers/data/', {listingG.name});

options.numPasses = 1000;
options.detailedStatFreq = 25000;
options.examplesFreq = 25000; 

end
