function [ hyperParams, options, wordMap, labelMap ] = SNLI(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, wordsource, dp, gc, lstminit)
% Configuration for our in-development corpus.
% See Defaults.m for parameter descriptions.

[hyperParams, options] = Defaults();
hyperParams.parensInSequences = false;
hyperParams.largeVocabMode = true;
hyperParams.loadWords = true;
hyperParams.trainWords = true;

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-pen', num2str(penult), '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(wordsource),...
    '-comp', num2str(composition), ...
    '-dp', num2str(dp), '-gc', num2str(gc),  '-lstminit', num2str(lstminit)];


hyperParams.LSTMinitType = lstminit;
hyperParams.dataPortion = dp;
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;

if wordsource == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif wordsource == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.'); 
elseif wordsource == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
end

hyperParams.useEmbeddingTransform = true;
hyperParams.topDepth = topDepth;
hyperParams.penultDim = penult;
hyperParams.lambda = lambda;
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;
hyperParams = CompositionSetup(hyperParams, composition);
hyperParams.useThirdOrderMerge = false;

if gc > 0
    hyperParams.clipGradients = true;
    hyperParams.maxGradNorm = gc;
end

if strcmp(dataflag, 'snli095-sick')
    wordMap = LoadWordMap('./sick-data/sick-snli0.95_words.txt');
    hyperParams.vocabName = 'ss095'; 

    hyperParams.numLabels = [3, 3];

    hyperParams.labels = {{'entailment', 'contradiction', 'neutral'},
                             {'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}};
    labelMap = cell(2, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));
    labelMap{2} = containers.Map(hyperParams.labels{2}, 1:length(hyperParams.labels{2}));

    hyperParams.trainFilenames = {'../data/snli_0.95_train_parsed.txt', ...
                                  './sick-data/SICK_train_parsed.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'./sick-data/SICK_trial_parsed.txt', ...
                                 '../data/snli_0.95_dev_parsed.txt'};

    hyperParams.labelIndices = [1, 2; 2, 1; 0, 0];
    hyperParams.testLabelIndices = [2, 1];
    hyperParams.trainingMultipliers = [1; mult];

elseif strcmp(dataflag, 'snli095-only')
    wordMap = LoadWordMap('./sick-data/sick-snli0.95_words.txt');
    hyperParams.vocabName = 'ss095'; 

    hyperParams.numLabels = [3];

    hyperParams.labels = {{'entailment', 'contradiction', 'neutral'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'../data/snli_0.95_train_parsed.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'../data/snli_0.95_dev_parsed.txt'};

    hyperParams.labelIndices = [1; 1; 1];
    hyperParams.testLabelIndices = [1];
    hyperParams.trainingMultipliers = [1];

elseif strcmp(dataflag, 'snlirc2-only')
    wordMap = LoadWordMap('../data/snlirc2_words.txt');
    hyperParams.vocabName = 'src2'; 

    hyperParams.numLabels = [3];

    hyperParams.labels = {{'entailment', 'contradiction', 'neutral'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'../data/snli_1.0rc2_train.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'../data/snli_1.0rc2_dev.txt'};

    hyperParams.labelIndices = [1; 1; 1];
    hyperParams.testLabelIndices = [1];
    hyperParams.trainingMultipliers = [1];

elseif strcmp(dataflag, 'snlirc3-only')
    wordMap = LoadWordMap('../data/snlirc3_words.txt');
    hyperParams.vocabName = 'src3'; 

    hyperParams.numLabels = [3];

    hyperParams.labels = {{'entailment', 'contradiction', 'neutral'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'../data/snli_1.0rc3_train.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'../data/snli_1.0rc3_dev.txt', '../data/snli_1.0rc3_test.txt'};

    hyperParams.labelIndices = [1, 1; 1, 1; 1, 1];
    hyperParams.testLabelIndices = [1, 1];
    hyperParams.trainingMultipliers = [1];

elseif strcmp(dataflag, 'snlirc3-only-short')
    wordMap = LoadWordMap('../data/snlirc3_words.txt');
    hyperParams.vocabName = 'src3'; 

    hyperParams.numLabels = [3];

    hyperParams.labels = {{'entailment', 'contradiction', 'neutral'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'../data/snli_1.0rc3_train_firsttenth.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'../data/snli_1.0rc3_dev.txt', '../data/snli_1.0rc3_test.txt'};

    hyperParams.labelIndices = [1, 1; 1, 1; 1, 1];
    hyperParams.testLabelIndices = [1, 1];
    hyperParams.trainingMultipliers = [1];

elseif strcmp(dataflag, 'snli095short-only')
    wordMap = LoadWordMap('./sick-data/sick-snli0.95_words.txt');
    hyperParams.vocabName = 'ss095'; 

    hyperParams.numLabels = [3];

    hyperParams.labels = {{'entailment', 'contradiction', 'neutral'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'../data/snli_0.95_train_words_parsed_short.txt'};    
    hyperParams.splitFilenames = {};    
    hyperParams.testFilenames = {'../data/snli_0.95_dev_words_parsed_short.txt'};

    hyperParams.labelIndices = [1; 1; 1];
    hyperParams.testLabelIndices = [1];
    hyperParams.trainingMultipliers = [1];

elseif strcmp(dataflag, 'dg-pre')
    hyperParams.numLabels = [3, 3, 2];

    hyperParams.labels = {{'entailment', 'contradiction', 'neutral'},
                             {'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    labelMap = cell(3, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));
    labelMap{2} = containers.Map(hyperParams.labels{2}, 1:length(hyperParams.labels{2}));
    labelMap{3} = containers.Map(hyperParams.labels{3}, 1:length(hyperParams.labels{3}));

    wordMap = LoadWordMap('sick-data/sick-snli0.95_words.txt');
    hyperParams.vocabName = 'ss095';

    hyperParams.trainFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_first600k.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_last1k.tsv',
                                 '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_first1k.tsv'};
    hyperParams.splitFilenames = {};

    hyperParams.labelIndices = [3, 0; 3, 3; 0, 0];
    hyperParams.testLabelIndices = [3, 3];
end

end
