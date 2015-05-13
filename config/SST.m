function [ hyperParams, options, wordMap, labelMap ] = SST(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, collo, latte, curr, slant, ccs)
% Configuration for experiments involving the SemEval SICK challenge and ImageFlickr 30k. 

[hyperParams, options] = Defaults();

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-p', num2str(penult), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(collo),...
    '-comp', num2str(composition), '-lattEv', num2str(latte), '-curr', num2str(curr), ...
    '-slant', num2str(slant), '-ccs', num2str(ccs) ];

hyperParams.sentenceClassificationMode = 1;
hyperParams.SSTMode = 1;

%%

hyperParams.latticeLocalCurriculum = curr;

hyperParams.connectionCostScale = ccs;

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
else
    hyperParams.latticeFirstPastThreshold = 0;
    hyperParams.latticeFirstPastHardMax = false;
end
    
%%

hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;

if collo == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif collo == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.'); 
elseif collo == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
end

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = 1;

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = topDepth;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step.
hyperParams.largeVocabMode = true;

% The dimensionality of the classifier extra layers.
hyperParams.penultDim = dim;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = 32;

% Apply dropout to the top feature vector of each tree, preserving activations
% with this probability. If this is set to 1, dropout is effectively not used.
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;

hyperParams = CompositionSetup(hyperParams, composition);

hyperParams.loadWords = true;

% How often (in steps) to report cost.
options.costFreq = 250;

% How often (in steps) to run on test data.
options.testFreq = 250;

% How often to report confusion matrices and connection accuracies. 
% Should be a multiple of testFreq.
options.detailedStatFreq = 250;

if strcmp(dataflag, 'sst-expanded')
    wordMap = InitializeMaps('./sst-data/sst-words.txt');
    hyperParams.vocabName = 'sst'; 

    hyperParams.numLabels = [5];

    hyperParams.labels = {{'0', '1', '2', '3', '4'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'./sst-data/train_expanded.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'./sst-data/dev.txt', './sst-data/train_sample.txt'};

    % Loading this data is fast, and the preprocessed file winds up huge.
    hyperParams.ignorePreprocessedFiles = true;

    hyperParams.labelCostMultipliers = [4.878182632, 2.433623131, 0.3014847996, 1.826731877, 3.980980277];

elseif strcmp(dataflag, 'sst')
    wordMap = InitializeMaps('./sst-data/sst-words.txt');
    hyperParams.vocabName = 'sst'; 

    hyperParams.numLabels = [5];

    hyperParams.labels = {{'0', '1', '2', '3', '4'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'./sst-data/train.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'./sst-data/dev.txt'};
end

%'TEMP'
%hyperParams.lineLimit = 50;
%hyperParams.loadWords = false;
%hyperParams.embeddingTransformDepth = 1;
%hyperParams.embeddingDim = 100;
%hyperParams.gpu = 0;
%hyperParams.largeVocabMode = 0;

end
