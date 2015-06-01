function [ hyperParams, options, wordMap, labelMap ] = Sick(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, wordsource, adi, conDim)
% Configuration for experiments involving the SemEval SICK challenge and DenotationGraph. 
% See Defaults.m for parameter descriptions.

[hyperParams, options] = Defaults();
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;
hyperParams.loadWords = true;
hyperParams.largeVocabMode = true;
hyperParams.trainWords = true;

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth), '-pen', num2str(penult), ...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-ws', num2str(wordsource),...
    '-adi', num2str(adi), '-comp', num2str(composition), ...
    '-cdim', num2str(conDim)];

if wordsource == 0
    hyperParams.loadWords = false;
elseif wordsource == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif wordsource == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The wordsourcebert and Weston-sourced vectors only come in dim 50.'); 
elseif wordsource == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
else
    hyperParams.vocabPath = ['../data/wordsource.scaled.' num2str(embDim) 'd.txt'];    
end

hyperParams.restartUpdateRuleInTransfer = adi;
hyperParams.transferSoftmax = true;
hyperParams.useEmbeddingTransform = true;
hyperParams.topDepth = topDepth;
hyperParams.penultDim = penult;
hyperParams.lambda = lambda;
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;
hyperParams.useEyes = true;
hyperParams = CompositionSetup(hyperParams, composition);

options.miniBatchSize = 32;

% Amount to upsample SICK data.
datamult = 8;

if strcmp(dataflag, 'sick-only-dev')
    wordMap = LoadWordMap('sick-data/sick_basic_words.txt');
    hyperParams.vocabName = 'sick_all';

    hyperParams.numLabels = [3];
   	hyperParams.labels = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}};
	labelMap = cell(1, 1);
	labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    hyperParams.trainFilenames = {'./sick-data/SICK_train_parsed.txt'};    
    hyperParams.testFilenames = {'./sick-data/SICK_trial_parsed.txt', ...
    				 './sick-data/SICK_trial_parsed_justneg.txt', ...
    				 './sick-data/SICK_trial_parsed_noneg.txt', ...
    				 './sick-data/SICK_trial_parsed_18plusparens.txt', ...
    				 './sick-data/SICK_trial_parsed_lt18_parens.txt'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'sick-only')
    % The number of labels.
    hyperParams.numLabels = [3];

    hyperParams.labels = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    wordMap = LoadWordMap('sick-data/sick_basic_words.txt');
    hyperParams.vocabName = 'sick_all';

    hyperParams.trainFilenames = {'./sick-data/SICK_train_parsed.txt'};
    hyperParams.testFilenames = {'./sick-data/SICK_trial_parsed.txt',...
                                 './sick-data/SICK_test_annotated_rearranged_parsed.txt'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'sick-only-transfer')
    % The number of labels.
    hyperParams.numLabels = [3];
    hyperParams.loadWords = 0;

    hyperParams.labels = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    wordMap = LoadWordMap('sick-data/sick-rc3_words.txt');
    hyperParams.sourceWordMap = LoadWordMap('../data/snlirc3_words.txt');

    hyperParams.vocabName = 'sick_all';

    hyperParams.trainFilenames = {'./sick-data/SICK_train_parsed.txt'};
    hyperParams.testFilenames = {'./sick-data/SICK_trial_parsed.txt',...
                                 './sick-data/SICK_test_annotated_rearranged_parsed.txt'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'sick2-only')
    % The number of labels.
    hyperParams.numLabels = [2];

    hyperParams.labels = {{'ENTAILMENT', 'NEUTRAL'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    wordMap = LoadWordMap('sick-data/sick_basic_words.txt');
    hyperParams.vocabName = 'sick_all';

    hyperParams.trainFilenames = {'./sick-data/SICK2_train_parsed.txt'};
    hyperParams.testFilenames = {'./sick-data/SICK2_trial_parsed.txt',...
                                 './sick-data/SICK2_test_annotated_rearranged_parsed.txt'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'sick2-only-transfer')
    % The number of labels.
    hyperParams.numLabels = [2];
    hyperParams.loadWords = 0;

    hyperParams.labels = {{'ENTAILMENT', 'NEUTRAL'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    wordMap = LoadWordMap('sick-data/sick-rc3_words.txt');
    hyperParams.sourceWordMap = LoadWordMap('../data/snlirc3_words.txt');

    hyperParams.vocabName = 'sick_all';

    hyperParams.trainFilenames = {'./sick-data/SICK2_train_parsed.txt'};
    hyperParams.testFilenames = {'./sick-data/SICK2_trial_parsed.txt',...
                                 './sick-data/SICK2_test_annotated_rearranged_parsed.txt'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'dg-only') 
    % The number of labels.
    hyperParams.numLabels = [2];

    hyperParams.labels = {{'ENTAILMENT', 'NONENTAILMENT'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    wordMap = LoadWordMap('sick-data/sick-snli0.95_words.txt');
    hyperParams.vocabName = 'ss095';

    hyperParams.trainFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_first600k.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_first1k.tsv',
                                 '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_last1k.tsv'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'dg-only-sst') 
    % The number of labels.
    hyperParams.numLabels = [2];

    hyperParams.labels = {{'ENTAILMENT', 'NONENTAILMENT'}};
    labelMap = cell(1, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    wordMap = LoadWordMap('sst-data/sst-words.txt');
    hyperParams.vocabName = 'sst';

    hyperParams.trainFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_first600k.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_first1k.tsv',
                                 '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_1-2wds_last1k.tsv'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'sick-plus-600k-ea-dev') 
    % The number of labels.
    hyperParams.numLabels = [3, 2];

    hyperParams.labels = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    labelMap = cell(2, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));
    labelMap{2} = containers.Map(hyperParams.labels{2}, 1:length(hyperParams.labels{2}));

    wordMap = LoadWordMap('sick-data/all_sick_plus_t4.txt');
    hyperParams.vocabName = 'aspt4';

    hyperParams.trainingMultipliers = [(datamult * 6); (datamult * 6); 1];

    hyperParams.trainFilenames = {'./sick-data/SICK_train_parsed_exactAlign.txt', ...
                     './sick-data/SICK_train_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'./sick-data/SICK_trial_parsed_exactAlign.txt', ...
                     './sick-data/SICK_trial_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.labelIndices = [1, 1, 2; 1, 1, 2; 0, 0, 0];
    hyperParams.testLabelIndices = [1, 1, 2];
elseif strcmp(dataflag, 'sick-plus-600k-dev') 
    % The number of labels.
    hyperParams.numLabels = [3, 2];

    hyperParams.labels = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    labelMap = cell(2, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));
    labelMap{2} = containers.Map(hyperParams.labels{2}, 1:length(hyperParams.labels{2}));

    wordMap = LoadWordMap('sick-data/all_sick_plus_t4.txt');
    hyperParams.vocabName = 'aspt4';

    hyperParams.trainingMultipliers = [(datamult * 12); 1];

    hyperParams.trainFilenames = {'./sick-data/SICK_train_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'./sick-data/SICK_trial_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.labelIndices = [1, 2; 1, 2; 0, 0];
    hyperParams.testLabelIndices = [1, 2];
elseif strcmp(dataflag, 'sick-plus-600k-ea') 
    % The number of labels.
    hyperParams.numLabels = [3, 2];

    hyperParams.labels = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    labelMap = cell(2, 1);
    labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));
    labelMap{2} = containers.Map(hyperParams.labels{2}, 1:length(hyperParams.labels{2}));

    wordMap = LoadWordMap('sick-data/all_sick_plus_t4.txt');
    hyperParams.vocabName = 'aspt4';

    hyperParams.trainingMultipliers = [(datamult * 6); (datamult * 6); (datamult * 6); (datamult * 6); 1];

    hyperParams.trainFilenames = {'./sick-data/SICK_train_parsed_exactAlign.txt', ...
                     './sick-data/SICK_train_parsed.txt', ...
                     './sick-data/SICK_trial_parsed_exactAlign.txt', ...
                     './sick-data/SICK_trial_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'./sick-data/SICK_test_annotated_rearranged_parsed_exactAlign.txt',...
                     './sick-data/SICK_test_annotated_rearranged_parsed.txt', ...
                     './sick-data/SICK_trial_parsed_exactAlign.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.labelIndices = [1, 1, 1, 1, 2; 1, 1, 1, 2, 0; 0, 0, 0, 0, 0];
    hyperParams.testLabelIndices = [1, 1, 1, 2];
elseif strcmp(dataflag, 'imageflickrshort')
    % The number of labels.
    hyperParams.numLabels = [2]; 

    hyperParams.labels = {{'ENTAILMENT', 'NONENTAILMENT'}};
	labelMap = cell(1, 1);
	labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

    wordMap = LoadWordMap('sick-data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4-2cl';

    hyperParams.splitFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/clean_parsed_entailment_pairs_first500.tsv', ...
    				 './sick-data/clean_parsed_entailment_pairs_second10k_first500.tsv'};
    hyperParams.trainFilenames = {};
end


end
