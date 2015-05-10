function [ hyperParams, options, wordMap, labelMap ] = ALCIR(expName, dataflag, dim, topDepth, penult, lambda, composition, slant)

[hyperParams, options] = Defaults();

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-td', num2str(topDepth), '-comp', num2str(composition),  '-sl', num2str(slant) ];

hyperParams.sentenceClassificationMode = 1;

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = 0;

if slant == -1
	hyperParams.connectionCostScale = 0;
elseif slant == -2
	hyperParams.dataPortion = 0.33;
else
	hyperParams.latticeSlant = slant;
end


hyperParams.parensInSequences = 1;

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = topDepth;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step.
hyperParams.largeVocabMode = false;

% The dimensionality of the classifier extra layers.
hyperParams.penultDim = dim;

% Regularization coefficient.
hyperParams.lambda = lambda;

hyperParams = CompositionSetup(hyperParams, composition);

hyperParams.loadWords = false;
hyperParams.trainWords = true;

wordMap = InitializeMaps('./alcir-data/ALCIR-words.txt');
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
