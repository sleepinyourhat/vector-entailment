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

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;
hyperParams.useEmbeddingTransform = et;
hyperParams.topDepth = topDepth;
hyperParams.penultDim = dim;
hyperParams.lambda = lambda;

hyperParams = CompositionSetup(hyperParams, composition);

wordMap = LoadWordMap('./sat-data/sat_words.txt');
hyperParams.vocabName = 'sat'; 

hyperParams.numLabels = [ 2 ];

hyperParams.labels = {{'SAT', 'UNSAT'}};
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

hyperParams.trainFilenames = {};    
hyperParams.splitFilenames = {'./sat-data/sat.txt'};    
hyperParams.testFilenames = {};

end
