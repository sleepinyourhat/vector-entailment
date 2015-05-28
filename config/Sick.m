function [ hyperParams, options, wordMap, labelMap ] = Sick(expName, dataflag, embDim, dim, topDepth, penult, lambda, composition, bottomDropout, topDropout, wordsource, parens, conDim)
% Configuration for experiments involving the SemEval SICK challenge and ImageFlickr 30k. 

[hyperParams, options] = Defaults();

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth), '-pen', num2str(penult), ...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-ws', num2str(wordsource),...
    '-par', num2str(parens), '-comp', num2str(composition), ...
    '-cdim', num2str(conDim)];


% The dimensionality of the word/phrase vectors. Currently fixed at 25 to match
% the GloVe vectors.
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;

if wordsource == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif wordsource == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The wordsourcebert and Weston-sourced vectors only come in dim 50.'); 
elseif wordsource == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
else
    hyperParams.vocabPath = ['../data/wordsource.scaled.' num2str(embDim) 'd.txt'];    
end

options.updateFn = @AdaDeltaUpdate;

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.useEmbeddingTransform = 1;

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = topDepth;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step.
hyperParams.largeVocabMode = true; % If we train words, go ahead and use it.

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% Apply dropout to the top feature vector of each tree, preserving activations
% with this probability. If this is set to 1, dropout is effectively not used.
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;

hyperParams.useEyes = 1;

hyperParams = CompositionSetup(hyperParams, composition);

hyperParams.loadWords = true;
hyperParams.trainWords = true;

hyperParams.fragmentData = false;

% How many examples to run before taking a parameter update step on the accumulated gradients.
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

% Temp testing method.
if conDim == -100
    hyperParams.lineLimit = 500;
    hyperParams.loadWords = 0;
end

end
