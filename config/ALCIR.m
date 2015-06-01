function [ hyperParams, options, wordMap, labelMap ] = ALCIR(expName, dataflag, dim, topDepth, penult, lambda, composition, slant)
% Configuration for description logic experiments.
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
    '-td', num2str(topDepth), '-comp', num2str(composition),  '-sl', num2str(slant) ];

if slant == -2
    hyperParams.randomEmbeddingIndices = [1, 2];
elseif slant == -3
    hyperParams.randomEmbeddingIndices = [1, 2];
    hyperParams.smallVecs = 1;
end

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

hyperParams.useEmbeddingTransform = 0;

if slant == 1
    hyperParams.latticeFirstPastThreshold = 0.5;
    hyperParams.latticeFirstPastHardMax = false;
elseif slant == 2
    hyperParams.latticeFirstPastThreshold = 0.75;
    hyperParams.latticeFirstPastHardMax = false;
elseif slant == 3
    hyperParams.latticeFirstPastThreshold = 0.9;
    hyperParams.latticeFirstPastHardMax = false;
elseif slant == 4
    hyperParams.latticeFirstPastThreshold = 0.5;
    hyperParams.latticeFirstPastHardMax = true;
elseif slant == 5
    hyperParams.latticeFirstPastThreshold = 0.9;
    hyperParams.latticeFirstPastHardMax = true;
elseif slant == 6 % Here and above works.
    hyperParams.latticeFirstPastThreshold = 0.95;
    hyperParams.latticeFirstPastHardMax = true;
elseif slant == 7
    hyperParams.latticeFirstPastThreshold = 0.98;
    hyperParams.latticeFirstPastHardMax = true;
elseif slant == -1
    hyperParams.latticeFirstPastThreshold = 0;
    hyperParams.latticeFirstPastHardMax = true;
    hyperParams.connectionCostScale = 0;
else    
    hyperParams.latticeFirstPastThreshold = 0;
    hyperParams.latticeFirstPastHardMax = false;
end

hyperParams.topDepth = topDepth;
hyperParams.penultDim = dim;
hyperParams.lambda = lambda;
hyperParams = CompositionSetup(hyperParams, composition);

wordMap = LoadWordMap('./alcir-data/ALCIR-words.txt');
hyperParams.vocabName = 'subj'; 

hyperParams.numLabels = [ 2 ];

hyperParams.labels = {{'satisfiable', 'unsatisfiable'}};
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

if strcmp(dataflag, 'cv') 
    hyperParams.trainFilenames = {};
    hyperParams.splitFilenames = {'./alcir-data/ALCIR-data.txt'};    
    hyperParams.testFilenames = {};
elseif strcmp(dataflag, 'long') 
    hyperParams.trainFilenames = {'./alcir-data/ALCIR-data.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'./alcir-data/ALCIR-data-long.txt'};
elseif strcmp(dataflag, 'parsetest') 
    hyperParams.trainFilenames = {};    
    hyperParams.splitFilenames = {'./alcir-data/ALCIR-data-parsesupervise-10k.txt'};    
    hyperParams.testFilenames = {};
end

end
