function [ hyperParams, options, wordMap, labelMap ] = SST(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, wordsource, latte, curr, slant, ccs)
% Configuration for Stanford Sentiment Treebank experiments.
% See Defaults.m for parameter descriptions.

[hyperParams, options] = Defaults();
hyperParams.sentenceClassificationMode = true;
hyperParams.SSTMode = true;
hyperParams.largeVocabMode = true;
hyperParams.loadWords = true;

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-p', num2str(penult), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(wordsource),...
    '-comp', num2str(composition), '-lattEv', num2str(latte), '-curr', num2str(curr), ...
    '-slant', num2str(slant), '-ccs', num2str(ccs) ];


hyperParams.latticeLocalCurriculum = curr;
hyperParams.connectionCostScale = ccs;
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;
hyperParams.useEmbeddingTransform = true;
hyperParams.topDepth = topDepth;
hyperParams.penultDim = penult;
hyperParams.lambda = lambda;
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;
hyperParams = CompositionSetup(hyperParams, composition);

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
    
if wordsource == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif wordsource == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.'); 
elseif wordsource == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
end

options.miniBatchSize = 32;

if strcmp(dataflag, 'sst-expanded')
    wordMap = LoadWordMap('./sst-data/sst-words.txt');
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
elseif strcmp(dataflag, 'sst-expanded-test')
    wordMap = LoadWordMap('./sst-data/sst-words.txt');

    hyperParams.vocabName = 'sst'; 

    hyperParams.numLabels = [5];

    hyperParams.labels = {{'0', '1', '2', '3', '4'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'./sst-data/train_expanded.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'./sst-data/dev.txt', './sst-data/test.txt', './sst-data/train_sample.txt'};

    % Loading this data is fast, and the preprocessed file winds up huge.
    hyperParams.ignorePreprocessedFiles = true;

    hyperParams.labelCostMultipliers = [4.878182632, 2.433623131, 0.3014847996, 1.826731877, 3.980980277];

elseif strcmp(dataflag, 'sst-expanded-transfer')
    wordMap = LoadWordMap('./sst-data/sst-snlirc2-transfer_words.txt');
    hyperParams.sourceWordMap = LoadWordMap('../data/snlirc2_words.txt');

    hyperParams.vocabName = 'sst-snlirc2'; 

    hyperParams.numLabels = [5];

    hyperParams.labels = {{'0', '1', '2', '3', '4'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'./sst-data/train_expanded.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'./sst-data/dev.txt', './sst-data/test.txt','./sst-data/train_sample.txt'};

    % Loading this data is fast, and the preprocessed file winds up huge.
    hyperParams.ignorePreprocessedFiles = true;

    hyperParams.labelCostMultipliers = [4.878182632, 2.433623131, 0.3014847996, 1.826731877, 3.980980277];

elseif strcmp(dataflag, 'sst')
    wordMap = LoadWordMap('./sst-data/sst-words.txt');
    hyperParams.vocabName = 'sst'; 

    hyperParams.numLabels = [5];

    hyperParams.labels = {{'0', '1', '2', '3', '4'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'./sst-data/train.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'./sst-data/dev.txt'};
end

end
