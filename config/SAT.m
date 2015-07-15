function [ hyperParams, options, wordMap, labelMap ] = SAT(expName, dataflag, dim, topDepth, lambda, composition, random, et)
% Configuration for SAT solving experiments.
% See Defaults.m for parameter descriptions.

[hyperParams, options] = Defaults();
hyperParams.sentenceClassificationMode = true;
hyperParams.parensInSequences = true;
hyperParams.largeVocabMode = false;
hyperParams.loadWords = false;
hyperParams.trainWords = true;

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-td', num2str(topDepth), '-comp', num2str(composition), '-random', num2str(random), '-et', num2str(et) ];

if random
    hyperParams.randomEmbeddingIndices = [1:99];
end

if et == -1
	options.miniBatchSize = 4;
	et = 1;
end

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;
hyperParams.useEmbeddingTransform = et;
hyperParams.topDepth = topDepth;
hyperParams.penultDim = dim;
hyperParams.lambda = lambda;

hyperParams = CompositionSetup(hyperParams, composition);

options.examplesFreq = 25000; 

wordMap = LoadWordMap('./sat-data/sat_words.txt');
hyperParams.vocabName = 'sat'; 

hyperParams.numLabels = [ 2 ];

hyperParams.labels = {{'SAT', 'UNSAT'}};
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));


if strcmp(dataflag, 'sat')
	hyperParams.trainFilenames = {};    
	hyperParams.splitFilenames = {'./sat-data/sat.txt'};    
	hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'sat-l')
	hyperParams.trainFilenames = {};    
	hyperParams.splitFilenames = {'./sat-data/sat_l.txt'};    
	hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'sat-4')
	hyperParams.trainFilenames = {};    
	hyperParams.splitFilenames = {'./sat-data/sat_4.txt'};    
	hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'sat-3-25')
	hyperParams.trainFilenames = {};    
	hyperParams.splitFilenames = {'./sat-data/sat_3-25.txt'};    
	hyperParams.testFilenames = {};
	hyperParams.lineLimit = 10000;
elseif strcmp(dataflag, 'sat-ls')
	hyperParams.trainFilenames = {};    
	hyperParams.splitFilenames = {'./sat-data/sat_l_shuf.txt'};    
	hyperParams.testFilenames = {};
end

end
