function [ hyperParams, options, wordMap, relationMap ] = Sick(dataflag, transDepth, penult, lambda, tot, mbs, lr, trainwords, frag, loadwords, fastEmb)
% Configuration for experiments involving the SemEval SICK challenge and ImageFlickr 30k. 

[hyperParams, options] = Defaults();

% The dimensionality of the word/phrase vectors. Currently fixed at 25 to match
% the GloVe vectors.
hyperParams.dim = 25;

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = transDepth;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step.
hyperParams.fastEmbed = fastEmb;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% Use NTN layers in place of NN layers.
hyperParams.useThirdOrder = tot;
hyperParams.useThirdOrderComparison = tot;

hyperParams.loadWords = loadwords;
hyperParams.trainWords = trainwords;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = mbs;

options.lr = lr;

if findstr(dataflag, 'sick-only')
    wordMap = ...
        InitializeMaps('sick_data/sick_words_t4.txt');
    hyperParams.vocabName = 'sot4'; 

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
elseif strcmp(dataflag, 'sick-plus-10k')
    % The number of relations.
    hyperParams.numRelations = [3 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}, {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(2, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = ...
        InitializeMaps('sick_data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4b';

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
                      '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_10k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_justneg.txt', ...
                     './sick_data/SICK_trial_parsed_noneg.txt', ...
                     './sick_data/SICK_trial_parsed_18plusparens.txt', ...
                     './sick_data/SICK_trial_parsed_lt18_parens.txt', ...
                     './sick_data/denotation_graph_training_subsample.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2, 0, 0, 0, 0; 1, 1, 1, 1, 1, 2; 0, 0, 0, 0, 0, 0];
    hyperParams.fragmentData = false;
    elseif strcmp(dataflag, 'sick-plus-100k')
    % The number of relations.
    hyperParams.numRelations = [3 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}, {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(2, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = ...
        InitializeMaps('sick_data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4b';

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
                      '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_justneg.txt', ...
                     './sick_data/SICK_trial_parsed_noneg.txt', ...
                     './sick_data/SICK_trial_parsed_18plusparens.txt', ...
                     './sick_data/SICK_trial_parsed_lt18_parens.txt', ...
                     './sick_data/denotation_graph_training_subsample.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2, 0, 0, 0, 0; 1, 1, 1, 1, 1, 2; 0, 0, 0, 0, 0, 0];
    hyperParams.fragmentData = false;
elseif strcmp(dataflag, 'sick-plus-600k')
    % The number of relations.
    hyperParams.numRelations = [3 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}, {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(2, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = ...
        InitializeMaps('sick_data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4b';

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
                      '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_justneg.txt', ...
                     './sick_data/SICK_trial_parsed_noneg.txt', ...
                     './sick_data/SICK_trial_parsed_18plusparens.txt', ...
                     './sick_data/SICK_trial_parsed_lt18_parens.txt', ...
                     './sick_data/denotation_graph_training_subsample.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2, 0, 0, 0, 0; 1, 1, 1, 1, 1, 2; 0, 0, 0, 0, 0, 0];
    hyperParams.fragmentData = false;
elseif strcmp(dataflag, 'sick-plus')
    % The number of relations.
    hyperParams.numRelations = [3 2];

	hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}, {'ENTAILMENT', 'NONENTAILMENT'}};
	relationMap = cell(2, 1);
	relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
	relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = ...
        InitializeMaps('sick_data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4b';

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

    hyperParams.trainFilenames = {'./sick_data/clean_parsed_entailment_pairs_second10k.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/clean_parsed_entailment_pairs_first500.tsv', ...
    				 './sick_data/clean_parsed_entailment_pairs_second10k_first500.tsv'};
    hyperParams.splitFilenames = {};
    hyperParams.fragmentData = frag;
end

end
