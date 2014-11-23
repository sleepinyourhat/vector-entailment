function [ hyperParams, options, wordMap, relationMap ] = Sick(expName, dataflag, embDim, dim, topDepth, penult, lambda, tot, summing, mbs, lr, trainwords, loadwords, bottomDropout, topDropout, datamult, rtemult, nlimult, collo, tensorScale, wordScale, eyes)
% Configuration for experiments involving the SemEval SICK challenge and ImageFlickr 30k. 

[hyperParams, options] = Defaults();

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth), '-tr', num2str(trainwords),...
    '-pen', num2str(penult), '-lr', num2str(lr), '-loadw', num2str(loadwords),...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(collo),...
    '-m', num2str(datamult), '-tsc', num2str(tensorScale), '-wsc', num2str(wordScale),...
    '-mb', num2str(mbs), '-e', num2str(eyes), '-rte', num2str(rtemult), '-nli', num2str(nlimult)];

% The dimensionality of the word/phrase vectors. Currently fixed at 25 to match
% the GloVe vectors.
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;

% Used to compute the bound on the range for RNTN parameter initialization.
hyperParams.initScale = tensorScale; % 1?

% The raw range bound on word vectors.
hyperParams.wordScale = wordScale; % 0.1?

hyperParams.useEyes = eyes;

if collo == 1
    hyperParams.vocabPath = ['../data/glove.6B.' num2str(embDim) 'd.txt'];
elseif collo == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.');  
else
    hyperParams.vocabPath = ['../data/collo.scaled.' num2str(embDim) 'd.txt'];    
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
hyperParams.fastEmbed = trainwords; % If we train words, go ahead and use it.

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% Apply dropout to the top feature vector of each tree, preserving activations
% with this probability. If this is set to 1, dropout is effectively not used.
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;

% Use NTN layers in place of NN layers.
hyperParams.useThirdOrder = tot;
hyperParams.useThirdOrderComparison = tot;

hyperParams.useSumming = summing;

hyperParams.loadWords = loadwords;
hyperParams.trainWords = trainwords;

hyperParams.fragmentData = false;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = mbs;

options.lr = lr;

if findstr(dataflag, 'sick-only')
    wordMap = ...
        InitializeMaps('sick_data/combined_words.txt');
    hyperParams.vocabName = 'sick_all'; 

    hyperParams.numRelations = 3;
   	hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt'};    
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
    				 './sick_data/SICK_trial_parsed_justneg.txt', ...
    				 './sick_data/SICK_trial_parsed_noneg.txt', ...
    				 './sick_data/SICK_trial_parsed_18plusparens.txt', ...
    				 './sick_data/SICK_trial_parsed_lt18_parens.txt'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'sick-plus-100')
    % The number of relations.
    hyperParams.numRelations = [3 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}, {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(2, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = ...
        InitializeMaps('sick_data/all_words.txt');
    hyperParams.vocabName = 'comwc';

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
                      '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_justneg.txt', ...
                     './sick_data/SICK_trial_parsed_noneg.txt', ...
                     './sick_data/SICK_trial_parsed_18plusparens.txt', ...
                     './sick_data/SICK_trial_parsed_lt18_parens.txt', ...
                     './sick_data/denotation_graph_training_subsample.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2, 0, 0, 0, 0; 1, 1, 1, 1, 1, 2; 0, 0, 0, 0, 0, 0];
elseif strcmp(dataflag, 'sick-plus-10k') 
    % The number of relations.
    hyperParams.numRelations = [3 7 2 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'#', '=', '>', '<', '|', '^', '_'},
                             {'ENTAILMENT', 'NONENTAILMENT'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(4, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));
    relationMap{3} = containers.Map(hyperParams.relations{3}, 1:length(hyperParams.relations{3}));
    relationMap{4} = containers.Map(hyperParams.relations{4}, 1:length(hyperParams.relations{4}));

    wordMap = InitializeMaps('sick_data/all_words.txt');
    hyperParams.vocabName = 'comwc';

    hyperParams.trainingMultipliers = [datamult; nlimult; rtemult; 1];

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
                      '../data/parsed_wcmac_data.txt', ...
                      '../data/parsed_rte.txt', ...
                      '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_10k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_justneg.txt', ...
                     './sick_data/SICK_trial_parsed_noneg.txt', ...
                     './sick_data/SICK_trial_parsed_18plusparens.txt', ...
                     './sick_data/SICK_trial_parsed_lt18_parens.txt', ...
                     './sick_data/denotation_graph_training_subsample.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2, 3, 4, 0, 0; 1, 1, 1, 1, 1, 4; 0, 0, 0, 0, 0, 0];
elseif strcmp(dataflag, 'sick-plus-100k') 
    % The number of relations.
    hyperParams.numRelations = [3 7 2 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'#', '=', '>', '<', '|', '^', '_'},
                             {'ENTAILMENT', 'NONENTAILMENT'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(4, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));
    relationMap{3} = containers.Map(hyperParams.relations{3}, 1:length(hyperParams.relations{3}));
    relationMap{4} = containers.Map(hyperParams.relations{4}, 1:length(hyperParams.relations{4}));

    wordMap = InitializeMaps('sick_data/all_words.txt');
    hyperParams.vocabName = 'comwc';

    hyperParams.trainingMultipliers = [datamult; nlimult; rtemult; 1];

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
                      '../data/parsed_wcmac_data.txt', ...
                      '../data/parsed_rte.txt', ...
                      '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_justneg.txt', ...
                     './sick_data/SICK_trial_parsed_noneg.txt', ...
                     './sick_data/SICK_trial_parsed_18plusparens.txt', ...
                     './sick_data/SICK_trial_parsed_lt18_parens.txt', ...
                     './sick_data/denotation_graph_training_subsample.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2, 3, 4, 0, 0; 1, 1, 1, 1, 1, 4; 0, 0, 0, 0, 0, 0];
elseif strcmp(dataflag, 'sick-plus-600k') 
    % The number of relations.
    hyperParams.numRelations = [3 7 2 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'#', '=', '>', '<', '|', '^', '_'},
                             {'ENTAILMENT', 'NONENTAILMENT'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(4, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));
    relationMap{3} = containers.Map(hyperParams.relations{3}, 1:length(hyperParams.relations{3}));
    relationMap{4} = containers.Map(hyperParams.relations{4}, 1:length(hyperParams.relations{4}));

    wordMap = InitializeMaps('sick_data/all_words.txt');
    hyperParams.vocabName = 'comwc';

    hyperParams.trainingMultipliers = [datamult; nlimult; rtemult; 1]; % [33, 40, 20, 1]
    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
                      '../data/parsed_wcmac_data.txt', ...
                      '../data/parsed_rte.txt', ...
                      '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_justneg.txt', ...
                     './sick_data/SICK_trial_parsed_noneg.txt', ...
                     './sick_data/SICK_trial_parsed_18plusparens.txt', ...
                     './sick_data/SICK_trial_parsed_lt18_parens.txt', ...
                     './sick_data/denotation_graph_training_subsample.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2, 3, 4, 0, 0; 1, 1, 1, 1, 1, 4; 0, 0, 0, 0, 0, 0];
elseif strcmp(dataflag, 'sick-plus')
    % The number of relations.
    hyperParams.numRelations = [3 2];

	hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}, {'ENTAILMENT', 'NONENTAILMENT'}};
	relationMap = cell(2, 1);
	relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
	relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = ...
        InitializeMaps('sick_data/all_words.txt');
    hyperParams.vocabName = 'comwc';

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
     				  '/scr/nlp/data/ImageFlickrEntailments/clean_parsed_entailment_pairs.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
    				 './sick_data/SICK_trial_parsed_justneg.txt', ...
    				 './sick_data/SICK_trial_parsed_noneg.txt', ...
    				 './sick_data/SICK_trial_parsed_18plusparens.txt', ...
    				 './sick_data/SICK_trial_parsed_lt18_parens.txt', ...
    				 './sick_data/denotation_graph_training_subsample.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2, 0, 0, 0, 0; 1, 1, 1, 1, 1, 2; 0, 0, 0, 0, 0, 0];
    hyperParams.fragmentData = true;
elseif strcmp(dataflag, 'imageflickr')
    % The number of relations.
    hyperParams.numRelations = 4; 

    hyperParams.relations = {{'ENTAILMENT', 'na', 'na2', 'NONENTAILMENT'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

    wordMap = InitializeMaps('sick_data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4b';

    hyperParams.trainFilenames = {'/scr/nlp/data/ImageFlickrEntailments/clean_parsed_entailment_pairs.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/clean_parsed_entailment_pairs_first500.tsv', ...
    				 './sick_data/clean_parsed_entailment_pairs_second10k_first500.tsv'};
    hyperParams.splitFilenames = {};
    hyperParams.fragmentData = true;
elseif strcmp(dataflag, 'imageflickrshort')
    % The number of relations.
    hyperParams.numRelations = 2; 

    hyperParams.relations = {{'ENTAILMENT', 'NONENTAILMENT'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

    wordMap = InitializeMaps('sick_data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4-2cl';

    hyperParams.splitFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_10k.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/clean_parsed_entailment_pairs_first500.tsv', ...
    				 './sick_data/clean_parsed_entailment_pairs_second10k_first500.tsv'};
    hyperParams.trainFilenames = {};
elseif strcmp(dataflag, 'imageflickrshort100')
    % The number of relations.
    hyperParams.numRelations = 2; 

    hyperParams.relations = {{'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(1, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

    wordMap = InitializeMaps('sick_data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4-2cl';

    hyperParams.splitFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100k.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/clean_parsed_entailment_pairs_first500.tsv', ...
                     './sick_data/clean_parsed_entailment_pairs_second10k_first500.tsv'};
    hyperParams.trainFilenames = {};
end

end
